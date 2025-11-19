import json
import random
from pathlib import Path
from collections import defaultdict


class DatasetValidator:
    REQUIRED_FIELDS = [
        'question_id',
        'dataset',
        'split',
        'question',
        'answer',
        'image_path',
        'scene_graph',
        'program',
        'metadata'
    ]
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.stats = defaultdict(int)
        self.errors = []
    
    def log(self, message):
        if self.verbose:
            print(f"  {message}")
    
    def validate_sample(self, sample):
        errors = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        if not isinstance(sample['question_id'], str):
            errors.append(f"question_id must be string, got {type(sample['question_id'])}")
        
        if not isinstance(sample['question'], str):
            errors.append(f"question must be string, got {type(sample['question'])}")
        
        if not isinstance(sample['answer'], str):
            errors.append(f"answer must be string, got {type(sample['answer'])}")
        
        if sample['dataset'] not in ['clevr', 'gqa']:
            errors.append(f"dataset must be 'clevr' or 'gqa', got '{sample['dataset']}'")
        
        project_root = Path(__file__).resolve().parent.parent
        image_path = project_root / sample['image_path']
        if not image_path.exists():
            errors.append(f"Image not found: {sample['image_path']}")
        
        if sample['scene_graph'] is not None:
            if not isinstance(sample['scene_graph'], dict):
                errors.append(f"scene_graph must be dict or null, got {type(sample['scene_graph'])}")
            else:
                if sample['dataset'] == 'clevr':
                    if 'objects' not in sample['scene_graph']:
                        errors.append("CLEVR scene_graph missing 'objects' field")
                elif sample['dataset'] == 'gqa':
                    if 'objects' not in sample['scene_graph']:
                        errors.append("GQA scene_graph missing 'objects' field")
        
        if sample['dataset'] == 'clevr':
            if sample['program'] is None:
                errors.append("CLEVR samples must have a program (not null)")
            elif not isinstance(sample['program'], list):
                errors.append(f"CLEVR program must be a list, got {type(sample['program'])}")
            else:
                for i, step in enumerate(sample['program']):
                    if not isinstance(step, dict):
                        errors.append(f"Program step {i} must be dict, got {type(step)}")
                    elif 'function' not in step:
                        errors.append(f"Program step {i} missing 'function' field")
        
        if sample['dataset'] == 'gqa':
            if sample['program'] is not None:
                errors.append("GQA samples should have program=null")
        
        if not isinstance(sample['metadata'], dict):
            errors.append(f"metadata must be dict, got {type(sample['metadata'])}")
        
        return len(errors) == 0, errors
    
    def validate_file(self, jsonl_path, num_samples=None):
        
        print()
        print(f"Validating: {jsonl_path}")
        print()
        
        if not jsonl_path.exists():
            print(f"ERROR: File not found: {jsonl_path}")
            return 0, 0, 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"Total samples in file: {total_lines}")
        
        if num_samples and num_samples < total_lines:
            sampled_indices = random.sample(range(total_lines), num_samples)
            lines_to_validate = [(i, lines[i]) for i in sampled_indices]
        else:
            print(f"Validating all {total_lines} samples...")
            lines_to_validate = list(enumerate(lines))
        
        valid_count = 0
        invalid_count = 0
        validation_errors = []
        
        for line_num, line in lines_to_validate:
            try:
                sample = json.loads(line.strip())
            
                is_valid, errors = self.validate_sample(sample)
                
                if is_valid:
                    valid_count += 1
                    self.log(f"Line {line_num + 1}: VALID")
                else:
                    invalid_count += 1
                    error_msg = f"X Line {line_num + 1}: INVALID"
                    print(f"    {error_msg}")
                    for error in errors:
                        print(f"    - {error}")
                    validation_errors.append((line_num + 1, errors))
            
            except json.JSONDecodeError as e:
                invalid_count += 1
                error_msg = f"X Line {line_num + 1}: JSON PARSE ERROR"
                print(f"  {error_msg}")
                print(f"    - {str(e)}")
                validation_errors.append((line_num + 1, [str(e)]))
            
            except Exception as e:
                invalid_count += 1
                error_msg = f"X Line {line_num + 1}: UNEXPECTED ERROR"
                print(f"    {error_msg}")
                print(f"    - {str(e)}")
                validation_errors.append((line_num + 1, [str(e)]))
        print()
        print(f"Validation Summary for {jsonl_path.name}")
        print()
        print(f"  Total validated: {len(lines_to_validate)}")
        print(f"  Valid samples: {valid_count} ({valid_count/len(lines_to_validate)*100:.2f}%)")
        print(f"  Invalid samples: {invalid_count} ({invalid_count/len(lines_to_validate)*100:.2f}%)")
        
        if invalid_count > 0:
            print(f"\n  First 5 errors:")
            for line_num, errors in validation_errors[:5]:
                print(f"    Line {line_num}:")
                for error in errors[:3]:  
                    print(f"      - {error}")
        print()
        
        return len(lines_to_validate), valid_count, invalid_count


def main():
    base_path = Path(__file__).resolve().parent
    converted_dir = base_path / "converted"
    num_samples = 100 
    
    if not converted_dir.exists():
        print(f"ERROR: Converted directory not found: {converted_dir}")
        print(f"       Please run convert_clevr.py and convert_gqa.py first.")
        return
    
    print()
    print("DATASET VALIDATOR")
    print()
    print(f"Converted directory: {converted_dir.absolute()}")
    print(f"Sample size: {num_samples} random samples per file")
    print()
    
    validator = DatasetValidator(verbose=False)
    
    files_to_validate = list(converted_dir.glob('*/*.jsonl'))
    
    if not files_to_validate:
        print(f"\nERROR: No JSONL files found in {converted_dir}")
        print(f"       Expected files like: clevr/clevr_train.jsonl, gqa/gqa_train.jsonl")
        return
    
    print(f"\nFound {len(files_to_validate)} JSONL files to validate:")
    for f in sorted(files_to_validate):
        print(f"  - {f.relative_to(base_path)}")

    overall_stats = {
        'total_validated': 0,
        'total_valid': 0,
        'total_invalid': 0
    }
    
    for file_path in sorted(files_to_validate):
        total, valid, invalid = validator.validate_file(file_path, num_samples=num_samples)
        
        overall_stats['total_validated'] += total
        overall_stats['total_valid'] += valid
        overall_stats['total_invalid'] += invalid
    
    print()
    print("OVERALL VALIDATION SUMMARY")
    print()
    print(f"  Files validated: {len(files_to_validate)}")
    print(f"  Total samples validated: {overall_stats['total_validated']}")
    
    if overall_stats['total_validated'] > 0:
        valid_pct = overall_stats['total_valid']/overall_stats['total_validated']*100
        invalid_pct = overall_stats['total_invalid']/overall_stats['total_validated']*100
        print(f"  Total valid: {overall_stats['total_valid']} ({valid_pct:.2f}%)")
        print(f"  Total invalid: {overall_stats['total_invalid']} ({invalid_pct:.2f}%)")
    
    print()
    print()
    if overall_stats['total_invalid'] == 0:
        print("\nAll validated samples are VALID!")
    else:
        print(f"\nX Found {overall_stats['total_invalid']} invalid samples")


if __name__ == '__main__':
    main()
