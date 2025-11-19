"""
Unified format:
  {
    "question_id": "CLEVR_train_000000",
    "dataset": "clevr",
    "split": "train",
    "question": "What is the color of the cube?",
    "answer": "red",
    "image_path": "datasets/clevr/images/train/CLEVR_train_000000.png",
    "image_id": image_filename.replace('.png', ''),
    "scene_graph": {...},
    "program": [...],
    "metadata": {...}
  }
"""   

import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def build_scene_lookup(scenes_data):
    scene_lookup = {}
    scenes = scenes_data.get('scenes', [])
    
    for scene in scenes:
        img_filename = scene.get('image_filename')
        if img_filename:
            scene_lookup[img_filename] = scene    
    return scene_lookup


def convert_clevr_split(questions_path, scenes_path, images_dir, output_path, split_name):
    questions_data = load_json(questions_path)
    scenes_data = load_json(scenes_path)

    scene_lookup = build_scene_lookup(scenes_data)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'success': 0,
        'missing_scene': 0,
        'missing_image': 0,
        'other_errors': 0
    }

    questions = questions_data.get('questions', [])
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for question in tqdm(questions, desc=f"Processing {split_name}"):
            stats['total'] += 1
            
            try:
                image_filename = question.get('image_filename')
                question_id = f"CLEVR_{split_name}_{question.get('image_index', 0):06d}_{question.get('question_index', 0)}"
                
                scene = scene_lookup.get(image_filename)
                if not scene:
                    stats['missing_scene'] += 1
                    continue
                
                image_path = f"datasets/clevr/images/{split_name}/{image_filename}"
                full_image_path = Path(images_dir) / image_filename
                
                if not full_image_path.exists():
                    stats['missing_image'] += 1
                    continue
                
                unified_sample = {
                    "question_id": question_id,
                    "dataset": "clevr",
                    "split": split_name,
                    "question": question.get('question', ''),
                    "answer": question.get('answer', ''),
                    "image_path": image_path,
                    "image_id": image_filename.replace('.png', ''),
                    "scene_graph": {
                        "objects": scene.get('objects', []),
                        "relationships": scene.get('relationships', {}),
                        "image_filename": image_filename,
                        "directions": scene.get('directions', {}),
                    },
                    "program": question.get('program', []),
                    "metadata": {
                        "image_index": question.get('image_index'),
                        "question_index": question.get('question_index'),
                        "question_family_index": question.get('question_family_index'),
                    }
                }

                out_file.write(json.dumps(unified_sample, ensure_ascii=False) + '\n')
                stats['success'] += 1
                
            except Exception as e:
                stats['other_errors'] += 1
                print(f"\nError processing question: {e}")
                continue

    print(f"Conversion Statistics for {split_name}:")
    print()
    print(f"  Total questions: {stats['total']}")
    print(f"  Successfully converted: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)")
    print(f"  Missing scene graphs: {stats['missing_scene']}")
    print(f"  Missing images: {stats['missing_image']}")
    print(f"  Other errors: {stats['other_errors']}")
    print(f"  Output: {output_path}")
    print()
    
    return stats


def main():
    base_path = Path(__file__).resolve().parent
    clevr_dir = base_path / "clevr"
    output_dir = base_path / "converted" / "clevr"
    splits = ["train", "val"]

    if not clevr_dir.exists():
        print(f"ERROR: CLEVR directory not found: {clevr_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("CLEVR Dataset Converter")
    print()
    print(f"CLEVR directory: {clevr_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Splits to convert: {', '.join(splits)}")
    print()

    overall_stats = defaultdict(int)

    for split in splits:
        questions_path = clevr_dir / 'questions' / f'CLEVR_{split}_questions.json'
        scenes_path = clevr_dir / 'scenes' / f'CLEVR_{split}_scenes.json'
        images_dir = clevr_dir / 'images' / split
        output_path = output_dir / f'clevr_{split}.jsonl'

        if not questions_path.exists():
            print(f"WARNING: Questions file not found: {questions_path}")
            continue
        if not scenes_path.exists():
            print(f"WARNING: Scenes file not found: {scenes_path}")
            continue
        if not images_dir.exists():
            print(f"WARNING: Images directory not found: {images_dir}")
            continue

        stats = convert_clevr_split(
            questions_path=questions_path,
            scenes_path=scenes_path,
            images_dir=images_dir,
            output_path=output_path,
            split_name=split
        )

        for key, value in stats.items():
            overall_stats[key] += value

    print()
    print("OVERALL CONVERSION STATISTICS")
    print()
    total = overall_stats["total"]
    success = overall_stats["success"]
    missing_scene = overall_stats["missing_scene"]
    missing_image = overall_stats["missing_image"]
    other_errors = overall_stats["other_errors"]

    pct = (success / total * 100) if total > 0 else 0.0

    print(f"    Total questions processed: {total}")
    print(f"    Successfully converted: {success} ({pct:.2f}%)")
    print(f"    Missing scene graphs: {missing_scene}")
    print(f"    Missing images: {missing_image}")
    print(f"    Other errors: {other_errors}")
    print()
    print("Conversion complete!")


if __name__ == '__main__':
    main()
