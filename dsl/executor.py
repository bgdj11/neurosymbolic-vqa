from typing import List, Dict, Any
from dsl.scene_graph import SceneGraph
from dsl.operators import get_operator


class ExecutionError(Exception):
    pass

class Executor:  
    def __init__(self, scene: SceneGraph, verbose: bool = False):
        self.scene = scene
        self.verbose = verbose
        self.results = []
        self.program = []
    
    def execute(self, program: List[Dict[str, Any]]) -> Any:
        if not program:
            raise ExecutionError("Cannot execute empty program")
        
        self.program = program
        self.results = []
        
        if self.verbose:
            print()
            print(f"EXECUTING PROGRAM ({len(program)} steps)")
            print()
        
        for step_idx, step in enumerate(program):
            try:
                result = self._execute_step(step_idx, step)
                self.results.append(result)
                
                if self.verbose:
                    self._print_step(step_idx, step, result)
                    
            except Exception as e:
                raise ExecutionError(
                    f"Error at step {step_idx} ({step.get('function', 'unknown')}): {e}"
                ) from e
        
        if self.verbose:
            print()
            print(f"FINAL RESULT: {self.results[-1]}")
            print()
        
        final_result = self.results[-1]
        if isinstance(final_result, bool):
            return 'yes' if final_result else 'no'
        
        return final_result
    
    def _execute_step(self, step_idx: int, step: Dict[str, Any]) -> Any:
        func_name = step.get('function')
        if not func_name:
            raise ExecutionError(f"Step {step_idx} missing 'function' field")
        try:
            operator = get_operator(func_name)
        except KeyError as e:
            raise ExecutionError(f"Unknown operator: {func_name}") from e

        inputs = []
        for input_idx in step.get('inputs', []):
            if input_idx >= len(self.results):
                raise ExecutionError(
                    f"Step {step_idx}: Invalid input reference {input_idx} "
                    f"(only {len(self.results)} previous results available)"
                )
            inputs.append(self.results[input_idx])
        
        value_inputs = step.get('value_inputs', [])
        
        side_inputs = step.get('side_inputs', [])

        args = [self.scene] + inputs

        if value_inputs:
            args.extend(value_inputs)
        elif side_inputs:
            args.extend(side_inputs)
        
        result = operator(*args)
        
        return result
    
    def _print_step(self, step_idx: int, step: Dict[str, Any], result: Any):
        func_name = step['function']
        inputs = step.get('inputs', [])
        value_inputs = step.get('value_inputs', [])
        side_inputs = step.get('side_inputs', [])
        
        input_strs = []
        for inp_idx in inputs:
            inp_val = self.results[inp_idx]
            input_strs.append(f"[{inp_idx}]={inp_val}")
        
        if value_inputs:
            input_strs.extend([f'"{v}"' for v in value_inputs])
        
        if side_inputs:
            input_strs.extend([f'"{v}"' for v in side_inputs])
        
        args_str = ", ".join(input_strs) if input_strs else ""
        
        print(f"[{step_idx}] {func_name}({args_str}) -> {result}")
    
    def get_intermediate_results(self) -> List[Any]:
        return self.results.copy()
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        if not self.program or not self.results:
            return []
        
        trace = []
        for i, (step, result) in enumerate(zip(self.program, self.results)):
            trace.append({
                'step_idx': i,
                'function': step.get('function'),
                'inputs': step.get('inputs', []),
                'value_inputs': step.get('value_inputs', []),
                'side_inputs': step.get('side_inputs', []),
                'result': result,
            })
        return trace


def execute_program(scene: SceneGraph, program: List[Dict[str, Any]], 
                   verbose: bool = False) -> Any:
    
    executor = Executor(scene, verbose=verbose)
    return executor.execute(program)