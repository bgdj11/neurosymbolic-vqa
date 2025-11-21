import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dsl.scene_graph import SceneGraph
from dsl.executor import Executor


def test_with_clevr_data():

    clevr_path = project_root / 'datasets' / 'converted' / 'clevr' / 'clevr_val.jsonl'
    
    if not clevr_path.exists():
        print(f"Dataset not found: {clevr_path}")
        return False
    
    print()
    print("TESTING EXECUTOR WITH REAL CLEVR DATA")
    print()
    
    success = 0
    missmatch = 0
    errors = 0
    limit = 10000

    with open(clevr_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            
            sample = json.loads(line)
            
            print()
            print(f"Sample {i+1}: {sample['question_id']}")
            print()
            print(f"Question: {sample['question']}")
            print(f"Expected answer: {sample['answer']}")
            print()
            
            try:
                scene = SceneGraph(sample['scene_graph'])
                
                executor = Executor(scene, verbose=False)
                result = executor.execute(sample['program'])

                result_str = str(result).lower()
                expected_str = str(sample['answer']).lower()
                
                if result_str == expected_str:
                    print(f"\nCORRECT! Got: {result}")
                    success += 1
                else:
                    print(f"\nMISMATCH! Got: {result}, Expected: {sample['answer']}")
                    missmatch += 1
                
            except Exception as e:
                print(f"\nERROR {e}")
                errors += 1
    
    print()
    print(f"RESULTS: {success} successful, {missmatch} mismatches, {errors} errors")
    print()
    
    if errors == 0:
        print("\nALL TESTS PASSED!")
        if missmatch == 0:
            print("ALL ANSWERS MATCHED EXPECTED RESULTS!")


if __name__ == '__main__':
    test_with_clevr_data()
