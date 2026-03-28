from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dsl.dsl_types import (
    DataType,
    OPERATOR_SIGNATURES,
    VALID_COLORS,
    VALID_SHAPES,
    VALID_SIZES,
    VALID_MATERIALS,
    VALID_RELATIONS,
)

# Operators that require exactly one value_input and what it must be from
_VALUE_INPUT_CONSTRAINTS: Dict[str, List[str]] = {
    'filter_color':    VALID_COLORS,
    'filter_shape':    VALID_SHAPES,
    'filter_size':     VALID_SIZES,
    'filter_material': VALID_MATERIALS,
}

# Operators that require exactly one side_input
_SIDE_INPUT_CONSTRAINTS: Dict[str, List[str]] = {}

# Operators that accept the value from either value_inputs OR side_inputs (executor reads both)
_EITHER_INPUT_CONSTRAINTS: Dict[str, List[str]] = {
    'relate': VALID_RELATIONS,
}


@dataclass
class ValidationError:
    step_idx: int
    step_function: str
    error_type: str   # 'unknown_operator' | 'type_mismatch' | 'invalid_value' | 'bad_ref' | 'structural'
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    inferred_types: List[Optional[DataType]]  # one per step, None if step not reached
    output_type: Optional[DataType]           # final step output type, None if invalid


class DSLValidator:

    def validate(self, program: List[Dict]) -> ValidationResult:
        if not isinstance(program, list) or len(program) == 0:
            err = ValidationError(
                step_idx=-1,
                step_function='',
                error_type='structural',
                message='Program must be a non-empty list of steps',
            )
            return ValidationResult(is_valid=False, errors=[err], inferred_types=[], output_type=None)

        structural_errors = self._check_structure(program)
        # If structural errors exist we still run type check but may get spurious errors;
        # return early only on truly unrecoverable issues (non-list program handled above).
        type_errors, inferred_types = self._check_types(program)

        all_errors = structural_errors + type_errors
        output_type = inferred_types[-1] if inferred_types and inferred_types[-1] is not None else None

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            inferred_types=inferred_types,
            output_type=output_type,
        )

    # Pass 1: structural check

    def _check_structure(self, program: List[Dict]) -> List[ValidationError]:
        errors: List[ValidationError] = []
        for i, step in enumerate(program):
            func_name = step.get('function', '') if isinstance(step, dict) else ''

            if not isinstance(step, dict):
                errors.append(ValidationError(i, '', 'structural', f'Step {i} is not a dict'))
                continue

            if not func_name:
                errors.append(ValidationError(i, '', 'structural', f'Step {i} missing "function" field'))
                continue

            # Check input references are backward-only and non-negative
            for ref in step.get('inputs', []):
                if not isinstance(ref, int):
                    errors.append(ValidationError(i, func_name, 'bad_ref',
                                                   f'Input reference must be int, got {type(ref).__name__}'))
                elif ref < 0:
                    errors.append(ValidationError(i, func_name, 'bad_ref',
                                                   f'Negative input reference {ref}'))
                elif ref >= i:
                    errors.append(ValidationError(i, func_name, 'bad_ref',
                                                   f'Forward reference: step {i} references step {ref} '
                                                   f'which has not been executed yet'))
        return errors

    # Pass 2: type-flow check + value input validation

    def _check_types(self, program: List[Dict]) -> Tuple[List[ValidationError], List[Optional[DataType]]]:
        errors: List[ValidationError] = []
        inferred: List[Optional[DataType]] = [None] * len(program)

        for i, step in enumerate(program):
            if not isinstance(step, dict):
                continue
            func_name = step.get('function', '')
            if not func_name:
                continue

            # Unknown operator
            sig = OPERATOR_SIGNATURES.get(func_name)
            if sig is None:
                errors.append(ValidationError(i, func_name, 'unknown_operator',
                                               f'Unknown operator "{func_name}"'))
                continue

            # Check input count matches signature
            step_inputs = step.get('inputs', [])
            expected_count = len(sig.inputs)
            if len(step_inputs) != expected_count:
                errors.append(ValidationError(i, func_name, 'structural',
                                               f'Operator "{func_name}" expects {expected_count} input(s), '
                                               f'got {len(step_inputs)}'))
            else:
                # Type-check each input
                for k, ref in enumerate(step_inputs):
                    if not isinstance(ref, int) or ref < 0 or ref >= i:
                        continue  # already reported in structural pass
                    actual = inferred[ref]
                    expected = sig.inputs[k]
                    if actual is not None and actual != expected:
                        errors.append(ValidationError(
                            i, func_name, 'type_mismatch',
                            f'Argument {k} of "{func_name}" expects {expected.value}, '
                            f'but step {ref} produces {actual.value}',
                        ))

            # Value input validation
            value_errors = self._check_value_inputs(
                i, func_name,
                step.get('value_inputs', []),
                step.get('side_inputs', []),
            )
            errors.extend(value_errors)

            # Record inferred output type (even if there were errors, for downstream steps)
            inferred[i] = sig.output

        return errors, inferred

    def _check_value_inputs(
        self,
        step_idx: int,
        func_name: str,
        value_inputs: List,
        side_inputs: List,
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if func_name in _VALUE_INPUT_CONSTRAINTS:
            valid_values = _VALUE_INPUT_CONSTRAINTS[func_name]
            if len(value_inputs) != 1:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'"{func_name}" requires exactly 1 value_input, '
                                               f'got {len(value_inputs)}'))
            elif value_inputs[0] not in valid_values:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'Invalid value "{value_inputs[0]}" for "{func_name}". '
                                               f'Valid values: {valid_values}'))

        if func_name in _SIDE_INPUT_CONSTRAINTS:
            valid_values = _SIDE_INPUT_CONSTRAINTS[func_name]
            if len(side_inputs) != 1:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'"{func_name}" requires exactly 1 side_input, '
                                               f'got {len(side_inputs)}'))
            elif side_inputs[0] not in valid_values:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'Invalid relation "{side_inputs[0]}" for "{func_name}". '
                                               f'Valid values: {valid_values}'))

        if func_name in _EITHER_INPUT_CONSTRAINTS:
            valid_values = _EITHER_INPUT_CONSTRAINTS[func_name]
            combined = value_inputs + side_inputs
            if len(combined) != 1:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'"{func_name}" requires exactly 1 value (in value_inputs or side_inputs), '
                                               f'got {len(combined)}'))
            elif combined[0] not in valid_values:
                errors.append(ValidationError(step_idx, func_name, 'invalid_value',
                                               f'Invalid value "{combined[0]}" for "{func_name}". '
                                               f'Valid values: {valid_values}'))

        return errors
