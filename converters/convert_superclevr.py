import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

PART_SHAPES = {
    'wheel', 'door', 'window', 'bumper', 'mirror', 'hood',
    'windshield', 'headlight', 'taillight', 'seat', 'handlebar',
    'fender', 'engine', 'roof', 'trunk', 'chassis', 'body',
}


def normalize_answer(answer) -> str:
    if isinstance(answer, bool):
        return 'yes' if answer else 'no'
    return str(answer)


def normalize_program(program: list) -> list:
    normalized = []
    for step in program:
        normalized.append({
            'function': step.get('type', step.get('function', '')),
            'inputs': step.get('inputs', []),
            'value_inputs': step.get('value_inputs', []),
            'side_inputs': step.get('side_inputs', []),
        })
    return normalized


def has_part_reference(question: str, program: list) -> bool:
    q_lower = question.lower()
    for part in PART_SHAPES:
        if part in q_lower:
            return True
    for step in program:
        for val in step.get('value_inputs', []) + step.get('side_inputs', []):
            if str(val).lower() in PART_SHAPES:
                return True
    return False


def convert_superclevr(questions_path, scenes_path, images_dir, output_path):
    print(f'Loading questions: {questions_path}')
    with open(questions_path, encoding='utf-8') as f:
        questions_data = json.load(f)

    print(f'Loading scenes: {scenes_path}')
    with open(scenes_path, encoding='utf-8') as f:
        scenes_data = json.load(f)

    scene_lookup = {
        s['image_filename']: s
        for s in scenes_data.get('scenes', [])
    }

    images_dir = Path(images_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)

    with open(output_path, 'w', encoding='utf-8') as out:
        for q in tqdm(questions_data.get('questions', []), desc='Super-CLEVR'):
            stats['total'] += 1
            img_filename = q.get('image_filename', '')
            question_text = q.get('question', '')
            program = q.get('program', [])

            if has_part_reference(question_text, program):
                stats['part_filtered'] += 1
                continue

            scene = scene_lookup.get(img_filename)
            if not scene:
                stats['missing_scene'] += 1
                continue

            if not (images_dir / img_filename).exists():
                stats['missing_image'] += 1

            qid = f"SuperCLEVR_{q.get('image_index', 0):06d}_{q.get('question_index', 0)}"

            sample = {
                'question_id': qid,
                'dataset': 'superclevr',
                'split': q.get('split', 'val'),
                'question': question_text,
                'answer': normalize_answer(q.get('answer', '')),
                'image_path': f"datasets/super_clevr/images/{img_filename}",
                'image_id': img_filename.replace('.png', ''),
                'scene_graph': {
                    'objects': scene.get('objects', []),
                    'relationships': scene.get('relationships', {}),
                    'image_filename': img_filename,
                    'directions': scene.get('directions', {}),
                },
                'program': normalize_program(program),
                'metadata': {
                    'image_index': q.get('image_index'),
                    'question_index': q.get('question_index'),
                },
            }
            out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['success'] += 1

    print(f'\nSuper-CLEVR conversion:')
    print(f'  Total:         {stats["total"]}')
    print(f'  Part-filtered: {stats["part_filtered"]}')
    print(f'  Missing scene: {stats["missing_scene"]}')
    print(f'  Missing image: {stats["missing_image"]} (GT tok works without images)')
    print(f'  Converted:     {stats["success"]}')
    print(f'  Output: {output_path}')
    return stats


def main():
    base = Path(__file__).resolve().parent.parent / "datasets"
    sc_dir = base / 'super_clevr'

    if not sc_dir.exists():
        print('ERROR: datasets/super_clevr/ not found.')
        return

    q_candidates = [
        sc_dir / 'superCLEVR_questions_30k.json',
        sc_dir / 'superCLEVR_questions_5.json',
        sc_dir / 'questions.json',
    ]
    s_candidates = [
        sc_dir / 'superCLEVR_scenes.json',
        sc_dir / 'scenes.json',
    ]

    q_path = next((p for p in q_candidates if p.exists()), None)
    s_path = next((p for p in s_candidates if p.exists()), None)

    if not q_path:
        print('ERROR: questions JSON not found.')
        return
    if not s_path:
        print('ERROR: scenes JSON not found.')
        return

    images_dir = sc_dir / 'images'
    out_path = base / 'converted' / 'superclevr' / 'superclevr_val.jsonl'

    convert_superclevr(q_path, s_path, images_dir, out_path)

    print('\nRun GT tok eval:')
    print('  python -m eval.run_eval --dataset superclevr --track gt --limit 1000 --batch 32')


if __name__ == '__main__':
    main()
