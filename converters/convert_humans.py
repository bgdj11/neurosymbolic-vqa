import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def convert_humans(questions_path, scenes_path, output_path):
    print(f'Loading CLEVR-Humans questions: {questions_path}')
    with open(questions_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f'Loading CLEVR val scenes: {scenes_path}')
    with open(scenes_path, encoding='utf-8') as f:
        scenes_data = json.load(f)

    scene_lookup = {
        s['image_filename']: s
        for s in scenes_data.get('scenes', [])
    }

    questions = data.get('questions', data.get('data', []))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)

    with open(output_path, 'w', encoding='utf-8') as out:
        for i, q in enumerate(tqdm(questions, desc='CLEVR-Humans')):
            stats['total'] += 1

            img_filename = (
                q.get('image_filename') or
                q.get('image') or
                f"CLEVR_val_{q.get('image_index', 0):06d}.png"
            )
            question_text = q.get('question', q.get('utterance', ''))
            answer = q.get('answer', q.get('response', ''))

            scene = scene_lookup.get(img_filename)
            if not scene:
                stats['missing_scene'] += 1
                continue

            qid = f"Humans_val_{i:06d}"

            sample = {
                'question_id': qid,
                'dataset': 'clevr_humans',
                'split': 'val',
                'question': question_text,
                'answer': str(answer),
                'image_path': f"datasets/clevr/images/val/{img_filename}",
                'image_id': img_filename.replace('.png', ''),
                'scene_graph': {
                    'objects': scene.get('objects', []),
                    'relationships': scene.get('relationships', {}),
                    'image_filename': img_filename,
                    'directions': scene.get('directions', {}),
                },
                'program': [],
                'metadata': {
                    'image_index': q.get('image_index'),
                    'source': 'human',
                },
            }
            out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['success'] += 1

    print(f'  Total: {stats["total"]}')
    print(f'  Converted: {stats["success"]}')
    print(f'  Missing scenes: {stats["missing_scene"]}')
    print(f'  Output: {output_path}')
    return stats


def main():
    base = Path(__file__).resolve().parent.parent / "datasets"

    scenes_path = base / 'clevr' / 'scenes' / 'CLEVR_val_scenes.json'
    if not scenes_path.exists():
        print(f'ERROR: {scenes_path} not found (CLEVR val scenes required).')
        return

    humans_dir = base / 'clevr_humans'
    if not humans_dir.exists():
        print('ERROR: datasets/clevr_humans/ not found.')
        print('Place the JSON file in: datasets/clevr_humans/')
        return

    q_candidates = [
        humans_dir / 'CLEVR-Humans-val.json',
        humans_dir / 'CLEVR_humans_val.json',
        humans_dir / 'CLEVR_v1.0_humans_val.json',
    ]
    q_path = next((p for p in q_candidates if p.exists()), None)
    if not q_path:
        print(f'ERROR: val JSON not found. Expected: {[p.name for p in q_candidates]}')
        print(f'Files in folder: {[p.name for p in humans_dir.glob("*.json")]}')
        return

    out_path = base / 'converted' / 'clevr_humans' / 'clevr_humans_val.jsonl'
    convert_humans(q_path, scenes_path, out_path)

    print('\nDone. Add to run_eval.py DATASETS:')
    print("  'clevr_humans': 'datasets/converted/clevr_humans/clevr_humans_val.jsonl'")


if __name__ == '__main__':
    main()
