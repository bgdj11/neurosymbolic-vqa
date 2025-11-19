"""
Unified format:
  {
    "question_id": "201234567",
    "dataset": "gqa",
    "split": "train",
    "question": "Is the person wearing a hat?",
    "answer": "yes",
    "image_path": "datasets/gqa/images/n12345.jpg",
    "image_id": image_id,
    "scene_graph": {...},
    "program": null,
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

def load_json_partial(path, limit=None):
    data = load_json(path)
    if limit and len(data) > limit:
        keys = list(data.keys())[:limit]
        data = {k: data[k] for k in keys}
    return data

def convert_gqa_split(questions_path, scenes_path, images_dir, output_path, split_name, scene_limit=None):
    print(f"Converting GQA {split_name} split")

    questions_data = load_json(questions_path)
    
    if scene_limit:
        print(f"    Note: Loading scene graphs with limit of {scene_limit} for memory efficiency")
    scenes_data = load_json_partial(scenes_path, limit=scene_limit) if scenes_path.exists() else {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'success': 0,
        'missing_scene': 0,
        'missing_image': 0,
        'other_errors': 0
    }

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for question_id, question_data in tqdm(questions_data.items(), desc=f"Processing {split_name}"):
            stats['total'] += 1
            try:
                image_id = question_data.get('imageId')

                scene = scenes_data.get(image_id)
                if not scene and scenes_path.exists():
                    stats['missing_scene'] += 1

                image_path = f"datasets/gqa/images/{image_id}.jpg"
                full_image_path = Path(images_dir) / f"{image_id}.jpg"

                if not full_image_path.exists():
                    stats['missing_image'] += 1
                    continue
                
                types = question_data.get('types', {})
                semantic_types = types.get('semantic', [])
                structural_types = types.get('structural', [])
                
                if isinstance(semantic_types, str):
                    semantic_types = [semantic_types]
                if isinstance(structural_types, str):
                    structural_types = [structural_types]
                
                scene_graph = None
                if scene:
                    scene_graph = {
                        "image_id": image_id,
                        "width": scene.get('width'),
                        "height": scene.get('height'),
                        "objects": scene.get('objects', {}),
                    }
                
                unified_sample = {
                    "question_id": question_id,
                    "dataset": "gqa",
                    "split": split_name,
                    "question": question_data.get('question', ''),
                    "answer": question_data.get('answer', ''),
                    "image_path": image_path,
                    "image_id": image_id,
                    "scene_graph": scene_graph,
                    "program": None,
                    "metadata": {
                        "image_id": image_id,
                        "full_answer": question_data.get('fullAnswer'),
                        "is_balanced": question_data.get('isBalanced', True),
                        "groups": question_data.get('groups', {}),
                        "semantic_types": semantic_types,
                        "structural_types": structural_types,
                    }
                }
                
                out_file.write(json.dumps(unified_sample, ensure_ascii=False) + '\n')
                stats['success'] += 1
                
            except Exception as e:
                stats['other_errors'] += 1
                print(f"\nError processing question {question_id}: {e}")
                continue

    print()
    print(f"Conversion Statistics for {split_name}:")
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
    gqa_dir = base_path / "gqa"
    output_dir = base_path / "processed"
    splits = ["train", "val"]
    scene_limit = None
    no_scenes = False 
    
    if not gqa_dir.exists():
        print(f"ERROR: GQA directory not found: {gqa_dir}")
        return
    
    print()
    print("GQA Dataset Converter")
    print()

    print(f"GQA directory: {gqa_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Splits to convert: {', '.join(splits)}")
    if no_scenes:
        print("Scene graphs: DISABLED (no_scenes=True)")
    elif scene_limit:
        print(f"Scene graph limit: {scene_limit}")
    print()

    overall_stats = defaultdict(int)

    for split in splits:
        questions_path = gqa_dir / "questions" / f"{split}_balanced_questions.json"
        scenes_path = gqa_dir / "scene_graphs" / f"{split}_sceneGraphs.json"
        images_dir = gqa_dir / "images"
        output_path = output_dir / f"gqa_{split}.jsonl"

        if not questions_path.exists():
            print(f"WARNING: Questions file not found: {questions_path}")
            continue
        
        if not scenes_path.exists() and not no_scenes:
            print(f"WARNING: Scene graphs file not found: {scenes_path}")
            print("     Continuing without scene graphs...")
        
        if not images_dir.exists():
            print(f"WARNING: Images directory not found: {images_dir}")
            continue
        
        stats = convert_gqa_split(
            questions_path=questions_path,
            scenes_path=scenes_path,
            images_dir=images_dir,
            output_path=output_path,
            split_name=split,
            scene_limit=scene_limit,
        )

        for key, value in stats.items():
            overall_stats[key] += value

    print()
    print("OVERALL CONVERSION STATISTICS")
    print()
    print(f"    Total questions processed: {overall_stats['total']}")
    print(f"    Successfully converted: {overall_stats['success']} ({overall_stats['success']/overall_stats['total']*100:.2f}%)")
    print(f"    Missing scene graphs: {overall_stats['missing_scene']}")
    print(f"    Missing images: {overall_stats['missing_image']}")
    print(f"    Other errors: {overall_stats['other_errors']}")
    print()
    print("Conversion complete!")

if __name__ == '__main__':
    main()
