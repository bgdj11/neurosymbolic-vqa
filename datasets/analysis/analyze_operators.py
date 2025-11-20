import json
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_clevr_programs(jsonl_path, max_samples=None):
    operators = Counter()
    value_inputs_examples = {}
    side_inputs_examples = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        if max_samples:
            lines = lines[:max_samples]
        
        for line in tqdm(lines, desc="Scanning programs"):
            sample = json.loads(line)
            program = sample.get('program', [])
            
            if not program:
                continue
            
            for step in program:
                func_name = step.get('function')
                if func_name:
                    operators[func_name] += 1
                    if 'value_inputs' in step and step['value_inputs']:
                        if func_name not in value_inputs_examples:
                            value_inputs_examples[func_name] = set()
                        for val in step['value_inputs']:
                            value_inputs_examples[func_name].add(val)
                    
                    if 'side_inputs' in step and step['side_inputs']:
                        if func_name not in side_inputs_examples:
                            side_inputs_examples[func_name] = set()
                        for val in step['side_inputs']:
                            side_inputs_examples[func_name].add(val)
    
    return operators, value_inputs_examples, side_inputs_examples


def print_analysis(operators, value_inputs, side_inputs):
    print()
    print("OPERATOR FREQUENCY")
    print()
    
    for op, count in operators.most_common():
        pct = (count / sum(operators.values())) * 100
        print(f"{op:25s} : {count:8d} occurrences ({pct:.2f}%)")
    
    print()
    print("VALUE INPUTS")
    print()
    
    for op in sorted(value_inputs.keys()):
        values = sorted(value_inputs[op])
        print(f"{op}:")
        print(f"  Values: {values}")
    
    print()
    print("SIDE INPUTS")
    print()
    
    for op in sorted(side_inputs.keys()):
        values = sorted(side_inputs[op])
        print(f"{op}:")
        print(f"  Values: {values}")
    
    print()
    print(f"TOTAL UNIQUE OPERATORS: {len(operators)}")
    print()


def compare_with_types_py():
    from dsl.dsl_types import get_all_operator_names
    
    defined_ops = set(get_all_operator_names())
    
    print()
    print("COMPARISON WITH types.py")
    print()
    
    print(f"Operators defined in types.py: {len(defined_ops)}")
    print(f"Defined operators: {sorted(defined_ops)}")
    
    return defined_ops


if __name__ == '__main__':

    clevr_val_path = project_root / 'datasets' / 'converted' / 'clevr' / 'clevr_val.jsonl'
    
    if not clevr_val_path.exists():
        print(f"ERROR: File not found: {clevr_val_path}")
        print("Please make sure the dataset is converted first.")
        sys.exit(1)
    
    operators, value_inputs, side_inputs = analyze_clevr_programs(clevr_val_path, max_samples=10000)

    print_analysis(operators, value_inputs, side_inputs)

    defined_ops = compare_with_types_py()

    found_ops = set(operators.keys())
    
    print()
    print("VALIDATION")
    print()
    
    missing_in_types = found_ops - defined_ops
    extra_in_types = defined_ops - found_ops
    
    if missing_in_types:
        print(f"MISSING in types.py (found in dataset but not defined):")
        for op in sorted(missing_in_types):
            print(f"    - {op}")
    else:
        print("All dataset operators are defined in types.py")
    
    print()
    
    if extra_in_types:
        print(f"EXTRA in types.py (defined but not found in scanned samples):")
        for op in sorted(extra_in_types):
            print(f"    - {op}")
        print("   (This is OK - they might appear in other samples)")
    
    print()