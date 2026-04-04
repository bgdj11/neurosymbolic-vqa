import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def convert_cogent_split(questions_path, scenes_path, output_path, split_name):
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)

    with open(output_path, 'w', encoding='utf-8') as out:
        for q in tqdm(questions_data.get('questions', []), desc=split_name):
            stats['total'] += 1
            img_filename = q.get('image_filename', '')
            scene = scene_lookup.get(img_filename)
            if not scene:
                stats['missing_scene'] += 1
                continue

            qid = f"CoGenT_{split_name}_{q.get('image_index', 0):06d}_{q.get('question_index', 0)}"

            sample = {
                'question_id': qid,
                'dataset': 'cogent',
                'split': split_name,
                'question': q.get('question', ''),
                'answer': q.get('answer', ''),
                'image_path': f"datasets/cogent/images/{split_name}/{img_filename}",
                'image_id': img_filename.replace('.png', ''),
                'scene_graph': {
                    'objects': scene.get('objects', []),
                    'relationships': scene.get('relationships', {}),
                    'image_filename': img_filename,
                    'directions': scene.get('directions', {}),
                },
                'program': q.get('program', []),
                'metadata': {
                    'image_index': q.get('image_index'),
                    'question_index': q.get('question_index'),
                    'condition': split_name,
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
    base = Path(__file__).resolve().parent
    cogent_dir = base / 'cogent'
    out_dir = base / 'converted' / 'cogent'

    if not cogent_dir.exists():
        print('ERROR: datasets/cogent/ not found.')
        print('Expected structure:')
        print('  datasets/cogent/questions/CLEVR_valB_questions.json')
        print('  datasets/cogent/scenes/CLEVR_valB_scenes.json')
        return

    q_candidates = [
        cogent_dir / 'questions' / 'CLEVR_CoGenT_val_B_questions.json',
        cogent_dir / 'questions' / 'CLEVR_valB_questions.json',
    ]
    s_candidates = [
        cogent_dir / 'scenes' / 'CLEVR_CoGenT_val_B_scenes.json',
        cogent_dir / 'scenes' / 'CLEVR_valB_scenes.json',
    ]

    q_path = next((p for p in q_candidates if p.exists()), None)
    s_path = next((p for p in s_candidates if p.exists()), None)

    if not q_path:
        print(f'ERROR: questions JSON not found in {cogent_dir}/questions/')
        print(f'Expected: {[p.name for p in q_candidates]}')
        return
    if not s_path:
        print(f'ERROR: scenes JSON not found in {cogent_dir}/scenes/')
        return

    convert_cogent_split(q_path, s_path, out_dir / 'cogent_valB.jsonl', 'valB')
    print('\nDone. Run:')
    print('  python -m eval.run_eval --dataset cogent_b --track gt --limit 1000 --batch 32')


if __name__ == '__main__':
    main()
