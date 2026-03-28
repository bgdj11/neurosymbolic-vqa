import pytest
from dsl.validator import DSLValidator, ValidationResult, ValidationError
from dsl.dsl_types import DataType, get_all_operator_names


def make_step(function, inputs=None, value_inputs=None, side_inputs=None):
    return {
        'function': function,
        'inputs': inputs or [],
        'value_inputs': value_inputs or [],
        'side_inputs': side_inputs or [],
    }


SCENE = make_step('scene')

SIMPLE_COUNT = [
    SCENE,
    make_step('count', inputs=[0]),
]

SIMPLE_FILTER_COUNT = [
    SCENE,
    make_step('filter_color', inputs=[0], value_inputs=['red']),
    make_step('count', inputs=[1]),
]


@pytest.fixture
def v():
    return DSLValidator()


# Structural tests

def test_empty_list_is_invalid(v):
    result = v.validate([])
    assert not result.is_valid
    assert any(e.error_type == 'structural' for e in result.errors)


def test_none_is_invalid(v):
    result = v.validate(None)
    assert not result.is_valid


def test_non_list_is_invalid(v):
    result = v.validate("scene | count")
    assert not result.is_valid


def test_step_not_dict_is_invalid(v):
    result = v.validate(["scene"])
    assert not result.is_valid
    assert any(e.error_type == 'structural' for e in result.errors)


def test_missing_function_field_is_invalid(v):
    result = v.validate([{'inputs': [], 'value_inputs': [], 'side_inputs': []}])
    assert not result.is_valid
    assert any(e.error_type == 'structural' for e in result.errors)


def test_forward_reference_is_invalid(v):
    # step 0 references step 1 which doesn't exist yet
    result = v.validate([make_step('filter_color', inputs=[1], value_inputs=['red'])])
    assert not result.is_valid
    assert any(e.error_type == 'bad_ref' for e in result.errors)


def test_self_reference_is_invalid(v):
    result = v.validate([make_step('count', inputs=[0])])
    assert not result.is_valid
    assert any(e.error_type == 'bad_ref' for e in result.errors)


def test_negative_reference_is_invalid(v):
    result = v.validate([SCENE, make_step('count', inputs=[-1])])
    assert not result.is_valid
    assert any(e.error_type == 'bad_ref' for e in result.errors)


def test_non_int_reference_is_invalid(v):
    result = v.validate([SCENE, make_step('count', inputs=['0'])])
    assert not result.is_valid
    assert any(e.error_type == 'bad_ref' for e in result.errors)


# Type-flow tests: valid programs

def test_single_scene_is_valid(v):
    result = v.validate([SCENE])
    assert result.is_valid
    assert result.output_type == DataType.OBJECT_SET


def test_scene_count_valid(v):
    result = v.validate(SIMPLE_COUNT)
    assert result.is_valid
    assert result.output_type == DataType.INTEGER


def test_scene_filter_count_valid(v):
    result = v.validate(SIMPLE_FILTER_COUNT)
    assert result.is_valid
    assert result.output_type == DataType.INTEGER


def test_exist_output_type_is_bool(v):
    result = v.validate([SCENE, make_step('exist', inputs=[0])])
    assert result.is_valid
    assert result.output_type == DataType.BOOL


def test_query_color_output_type_is_attribute(v):
    result = v.validate([
        SCENE,
        make_step('unique', inputs=[0]),
        make_step('query_color', inputs=[1]),
    ])
    assert result.is_valid
    assert result.output_type == DataType.ATTRIBUTE


def test_relate_valid(v):
    result = v.validate([
        SCENE,
        make_step('relate', inputs=[0], side_inputs=['left']),
        make_step('count', inputs=[1]),
    ])
    assert result.is_valid


def test_intersect_valid(v):
    result = v.validate([
        SCENE,
        make_step('filter_color', inputs=[0], value_inputs=['red']),
        make_step('filter_shape', inputs=[0], value_inputs=['cube']),
        make_step('intersect', inputs=[1, 2]),
        make_step('count', inputs=[3]),
    ])
    assert result.is_valid
    assert result.output_type == DataType.INTEGER


def test_equal_color_valid(v):
    result = v.validate([
        SCENE,
        make_step('unique', inputs=[0]),
        make_step('query_color', inputs=[1]),
        make_step('scene'),
        make_step('unique', inputs=[3]),
        make_step('query_color', inputs=[4]),
        make_step('equal_color', inputs=[2, 5]),
    ])
    assert result.is_valid
    assert result.output_type == DataType.BOOL


def test_less_than_valid(v):
    result = v.validate([
        SCENE,
        make_step('filter_color', inputs=[0], value_inputs=['red']),
        make_step('count', inputs=[1]),
        make_step('filter_color', inputs=[0], value_inputs=['blue']),
        make_step('count', inputs=[3]),
        make_step('less_than', inputs=[2, 4]),
    ])
    assert result.is_valid
    assert result.output_type == DataType.BOOL


def test_same_color_valid(v):
    result = v.validate([
        SCENE,
        make_step('same_color', inputs=[0]),
        make_step('count', inputs=[1]),
    ])
    assert result.is_valid


def test_inferred_types_length_matches_program(v):
    result = v.validate(SIMPLE_FILTER_COUNT)
    assert len(result.inferred_types) == len(SIMPLE_FILTER_COUNT)


def test_output_type_is_last_step(v):
    result = v.validate(SIMPLE_FILTER_COUNT)
    assert result.output_type == result.inferred_types[-1]


# Type-flow tests: type mismatches

def test_type_mismatch_bool_into_object_set(v):
    # exist → BOOL, then filter_color expects OBJECT_SET
    result = v.validate([
        SCENE,
        make_step('exist', inputs=[0]),
        make_step('filter_color', inputs=[1], value_inputs=['red']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'type_mismatch' for e in result.errors)


def test_type_mismatch_integer_into_attribute_slot(v):
    # count → INTEGER, equal_color expects ATTRIBUTE, ATTRIBUTE
    result = v.validate([
        SCENE,
        make_step('count', inputs=[0]),
        make_step('count', inputs=[0]),
        make_step('equal_color', inputs=[1, 2]),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'type_mismatch' for e in result.errors)


def test_wrong_input_count(v):
    # intersect needs 2 inputs, give it 1
    result = v.validate([
        SCENE,
        make_step('intersect', inputs=[0]),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'structural' for e in result.errors)


def test_wrong_input_count_too_many(v):
    # count needs 1 input, give it 2
    result = v.validate([
        SCENE,
        make_step('filter_color', inputs=[0], value_inputs=['red']),
        make_step('count', inputs=[0, 1]),
    ])
    assert not result.is_valid


# Value input validation tests

def test_invalid_color_value(v):
    result = v.validate([
        SCENE,
        make_step('filter_color', inputs=[0], value_inputs=['pink']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_invalid_shape_value(v):
    result = v.validate([
        SCENE,
        make_step('filter_shape', inputs=[0], value_inputs=['triangle']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_invalid_size_value(v):
    result = v.validate([
        SCENE,
        make_step('filter_size', inputs=[0], value_inputs=['medium']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_invalid_material_value(v):
    result = v.validate([
        SCENE,
        make_step('filter_material', inputs=[0], value_inputs=['wood']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_invalid_relation_value(v):
    result = v.validate([
        SCENE,
        make_step('relate', inputs=[0], side_inputs=['diagonal']),
    ])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_all_valid_colors_accepted(v):
    for color in ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']:
        result = v.validate([SCENE, make_step('filter_color', inputs=[0], value_inputs=[color])])
        assert result.is_valid, f"Color '{color}' should be valid"


def test_all_valid_shapes_accepted(v):
    for shape in ['cube', 'sphere', 'cylinder']:
        result = v.validate([SCENE, make_step('filter_shape', inputs=[0], value_inputs=[shape])])
        assert result.is_valid, f"Shape '{shape}' should be valid"


def test_all_valid_relations_accepted(v):
    for rel in ['left', 'right', 'front', 'behind']:
        result = v.validate([SCENE, make_step('relate', inputs=[0], side_inputs=[rel])])
        assert result.is_valid, f"Relation '{rel}' should be valid"


def test_missing_value_input_for_filter(v):
    result = v.validate([SCENE, make_step('filter_color', inputs=[0], value_inputs=[])])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


def test_missing_side_input_for_relate(v):
    result = v.validate([SCENE, make_step('relate', inputs=[0], side_inputs=[])])
    assert not result.is_valid
    assert any(e.error_type == 'invalid_value' for e in result.errors)


# Unknown operator

def test_unknown_operator_name(v):
    result = v.validate([make_step('filter_colours')])
    assert not result.is_valid
    assert any(e.error_type == 'unknown_operator' for e in result.errors)


def test_all_26_operators_recognized(v):
    all_ops = get_all_operator_names()
    assert len(all_ops) > 0, "No operators found in dsl_types"
    for op_name in all_ops:
        # Each known operator should not trigger 'unknown_operator'
        result = v.validate([make_step(op_name)])
        assert not any(e.error_type == 'unknown_operator' for e in result.errors), \
            f"Operator '{op_name}' was flagged as unknown"


# Integration: CLEVR val set smoke test

@pytest.mark.clevr
def test_all_clevr_val_programs_pass_validation():
    #pytest tests/test_dsl_validator.py -m clevr

    import json
    from pathlib import Path

    val_path = Path('datasets/converted/clevr/clevr_val.jsonl')
    if not val_path.exists():
        pytest.skip(f"CLEVR val file not found: {val_path}")

    validator = DSLValidator()
    total = valid = 0

    with open(val_path) as f:
        for line in f:
            sample = json.loads(line)
            program = sample.get('program')
            if program is None:
                continue
            total += 1
            result = validator.validate(program)
            if result.is_valid:
                valid += 1
            else:
                # Print first failure for diagnosis
                if total - valid == 1:
                    print(f"\nFirst invalid program (question_id={sample['question_id']}):")
                    for e in result.errors:
                        print(f"  step {e.step_idx} [{e.step_function}] {e.error_type}: {e.message}")

    pass_rate = valid / total if total > 0 else 0
    print(f"\nCLEVR val: {valid}/{total} programs valid ({pass_rate:.1%})")
    assert pass_rate >= 0.99, f"Expected >=99% valid programs, got {pass_rate:.1%}"
