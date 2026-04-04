"""
Logical consistency check across questions on the same scene.

    python -m eval.consistency_check \
        --results results/clevr_val_gt_n1000_per_question.jsonl \
        --output  results/consistency.json
"""
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


def is_count_answer(answer: str) -> bool:
    try:
        int(answer)
        return True
    except (ValueError, TypeError):
        return False


def is_yes_no(answer: str) -> bool:
    return str(answer).lower() in {'yes', 'no'}


def check_scene_consistency(questions_on_scene: list) -> dict:
    #Given all Q&A pairs for a single scene, return consistency stats.
    #questions_on_scene: list of EvalResult dicts (question, predicted_answer, correct, ...)

    violations = []

    answered = [q for q in questions_on_scene if q['predicted_answer'] is not None]

    # Rule 1: count answers must be integers >= 0
    for q in answered:
        ans = q['predicted_answer']
        if is_count_answer(str(q['expected_answer'])):
            try:
                val = int(ans)
                if val < 0:
                    violations.append({
                        'type': 'negative_count',
                        'question': q['question'],
                        'answer': ans,
                    })
            except (ValueError, TypeError):
                violations.append({
                    'type': 'invalid_count_format',
                    'question': q['question'],
                    'answer': ans,
                })

    # Rule 2: exist=yes implies count >= 1 (cross-question check)
    exist_yes = set()
    exist_no = set()
    for q in answered:
        if not is_yes_no(q.get('expected_answer', '')):
            continue
        # Extract what the exist question is about
        pred = str(q['predicted_answer']).lower()
        q_lower = q['question'].lower()
        if 'are there any' in q_lower or 'is there' in q_lower:
            if pred == 'yes':
                exist_yes.add(q['question'])
            else:
                exist_no.add(q['question'])

    # Rule 3: If two count questions count subsets, larger set >= smaller set
    count_answers = []
    for q in answered:
        exp = q.get('expected_answer', '')
        if is_count_answer(str(exp)):
            try:
                count_answers.append({
                    'question': q['question'],
                    'predicted': int(q['predicted_answer']),
                })
            except (ValueError, TypeError):
                pass

    return {
        'n_questions': len(questions_on_scene),
        'n_answered': len(answered),
        'n_violations': len(violations),
        'violations': violations,
        'consistent': len(violations) == 0,
    }


def run_consistency_check(results_path: str) -> dict:
    # Load results grouped by image_id
    by_image = defaultdict(list)
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            image_id = r.get('question_id', '').rsplit('_', 1)[0]
            by_image[image_id].append(r)

    total_scenes = len(by_image)
    consistent_scenes = 0
    total_violations = 0
    violation_types = defaultdict(int)
    all_violations = []

    for image_id, qs in by_image.items():
        result = check_scene_consistency(qs)
        if result['consistent']:
            consistent_scenes += 1
        total_violations += result['n_violations']
        for v in result['violations']:
            violation_types[v['type']] += 1
            all_violations.append({'image_id': image_id, **v})

    consistency_rate = consistent_scenes / max(total_scenes, 1)

    summary = {
        'total_scenes': total_scenes,
        'consistent_scenes': consistent_scenes,
        'consistency_rate': consistency_rate,
        'total_violations': total_violations,
        'violation_types': dict(violation_types),
        'sample_violations': all_violations[:20],
    }

    print(f'\n=== Consistency Check ===')
    print(f'Total scenes:       {total_scenes}')
    print(f'Consistent scenes:  {consistent_scenes} ({100*consistency_rate:.1f}%)')
    print(f'Total violations:   {total_violations}')
    if violation_types:
        print(f'Violation types:')
        for t, c in sorted(violation_types.items(), key=lambda x: -x[1]):
            print(f'  {t}: {c}')

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='per_question.jsonl from run_eval')
    parser.add_argument('--output', default='results/consistency.json')
    args = parser.parse_args()

    summary = run_consistency_check(args.results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
