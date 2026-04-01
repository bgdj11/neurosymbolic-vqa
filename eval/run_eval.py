import argparse
import json
import time
from pathlib import Path
from typing import Optional

from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph
from eval.metrics import EvalResult, categorize_question, compute_summary
from nl2dsl.infer import NL2DSLModel

DATASETS = {
    'clevr_val':  'datasets/converted/clevr/clevr_val.jsonl',
    'clevr_train': 'datasets/converted/clevr/clevr_train.jsonl',
    'cogent_b':   'datasets/converted/cogent/cogent_valB.jsonl',
}


def run_gt_tok(
    jsonl_path: str,
    model: NL2DSLModel,
    limit: Optional[int],
    batch_size: int,
) -> list:
    samples = []
    with open(jsonl_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))

    questions = [s['question'] for s in samples]
    print(f'Translating {len(questions)} questions (batch_size={batch_size})...')
    t0 = time.time()
    programs = model.predict_batch(questions, batch_size=batch_size)
    print(f'NL2DSL done in {time.time()-t0:.1f}s')

    results = []
    for sample, program in zip(samples, programs):
        cat = categorize_question(sample['question'])

        if program is None:
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='gt',
                program_status='failed',
                execution_error=None,
            ))
            continue

        try:
            sg = SceneGraph(sample['scene_graph'])
            result = Executor(sg).execute(program)
            predicted = str(result).lower()
            correct = predicted == str(sample['answer']).lower()
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=predicted,
                correct=correct,
                category=cat,
                track='gt',
                program_status='valid',
                execution_error=None,
            ))
        except ExecutionError as e:
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='gt',
                program_status='valid',
                execution_error=str(e),
            ))

    return results


def run_detected_tok(
    jsonl_path: str,
    model: NL2DSLModel,
    limit: Optional[int],
    batch_size: int,
) -> list:
    from scene_graph.builder import SceneGraphBuilder
    builder = SceneGraphBuilder()

    samples = []
    with open(jsonl_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))

    # Build detected scene graphs (cache by image path)
    image_cache = {}
    print(f'Building detected scene graphs for {len(samples)} samples...')
    for i, sample in enumerate(samples):
        img_path = sample['image_path']
        if img_path not in image_cache:
            if not Path(img_path).exists():
                image_cache[img_path] = None
            else:
                try:
                    image_cache[img_path] = builder.build(img_path)
                except Exception:
                    image_cache[img_path] = None
        if (i + 1) % 50 == 0:
            cached = sum(1 for v in image_cache.values() if v is not None)
            print(f'  {i+1}/{len(samples)} samples, {len(image_cache)} unique images ({cached} built)')

    # NL2DSL batch
    questions = [s['question'] for s in samples]
    print('Translating questions...')
    programs = model.predict_batch(questions, batch_size=batch_size)

    results = []
    for sample, program in zip(samples, programs):
        cat = categorize_question(sample['question'])
        detected_sg = image_cache.get(sample['image_path'])

        if program is None:
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='detected',
                program_status='failed',
                execution_error=None,
            ))
            continue

        if detected_sg is None:
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='detected',
                program_status='valid',
                execution_error='image_not_found',
            ))
            continue

        try:
            sg = SceneGraph(detected_sg)
            result = Executor(sg).execute(program)
            predicted = str(result).lower()
            correct = predicted == str(sample['answer']).lower()
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=predicted,
                correct=correct,
                category=cat,
                track='detected',
                program_status='valid',
                execution_error=None,
            ))
        except ExecutionError as e:
            results.append(EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='detected',
                program_status='valid',
                execution_error=str(e),
            ))

    return results


def save_results(results, summary, output_dir: str, tag: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_q_path = output_dir / f'{tag}_per_question.jsonl'
    with open(per_q_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + '\n')

    summary_path = output_dir / f'{tag}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary.__dict__, f, indent=2)

    print(f'Saved: {per_q_path}')
    print(f'Saved: {summary_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr_val', choices=list(DATASETS.keys()))
    parser.add_argument('--track',   default='gt', choices=['gt', 'detected'])
    parser.add_argument('--model',   default='models/t5-nl2dsl-final')
    parser.add_argument('--limit',   type=int, default=None)
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--output',  default='results')
    args = parser.parse_args()

    jsonl_path = DATASETS[args.dataset]
    if not Path(jsonl_path).exists():
        print(f'Dataset not found: {jsonl_path}')
        return

    print(f'Loading model from {args.model}...')
    model = NL2DSLModel(args.model)

    tag = f'{args.dataset}_{args.track}'
    if args.limit:
        tag += f'_n{args.limit}'

    if args.track == 'gt':
        results = run_gt_tok(jsonl_path, model, args.limit, args.batch)
    else:
        results = run_detected_tok(jsonl_path, model, args.limit, args.batch)

    summary = compute_summary(results, track=args.track, dataset=args.dataset)
    summary.print()
    save_results(results, summary, args.output, tag)


if __name__ == '__main__':
    main()
