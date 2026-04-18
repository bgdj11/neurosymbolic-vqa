"""
    python eval/run_llm_eval.py --dataset clevr_val --limit 1000
    python eval/run_llm_eval.py --dataset clevr_humans --limit 1000
    python eval/run_llm_eval.py --dataset clevr_val --llm-model llama3.2:1b --limit 1000
"""
import argparse
import json
import time
from pathlib import Path

from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph
from eval.metrics import EvalResult, categorize_question, compute_summary
from nl2dsl.llm_translator import LLMTranslator

DATASETS = {
    'clevr_val': 'datasets/converted/clevr/clevr_val.jsonl',
    'clevr_humans': 'datasets/converted/clevr_humans/clevr_humans_val.jsonl',
    'cogent_b': 'datasets/converted/cogent/cogent_valB.jsonl',
    'superclevr': 'datasets/converted/superclevr/superclevr_val.jsonl',
}


def run_llm_gt_tok(jsonl_path, model, limit, checkpoint_path):
    done_ids = {}
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done_ids[r['question_id']] = r
        print(f'Resuming: {len(done_ids)} already done.')

    samples = []
    with open(jsonl_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            s = json.loads(line)
            if s['question_id'] not in done_ids:
                samples.append(s)

    total = len(done_ids) + len(samples)
    print(f'Evaluating {len(samples)} questions with LLM (sequential)...')

    try:
        from tqdm import tqdm
        bar = tqdm(samples, desc='LLM eval')
    except ImportError:
        bar = samples

    ckpt_file = open(checkpoint_path, 'a', encoding='utf-8')
    results_new = []
    t0 = time.time()
    correct_so_far = 0

    for i, sample in enumerate(bar):
        program = model.predict_program(sample['question'])
        cat = categorize_question(sample['question'])

        if program is None:
            r = EvalResult(
                question_id=sample['question_id'],
                question=sample['question'],
                expected_answer=sample['answer'],
                predicted_answer=None,
                correct=False,
                category=cat,
                track='gt',
                program_status='failed',
                execution_error=None,
            )
        else:
            try:
                sg = SceneGraph(sample['scene_graph'])
                predicted = str(Executor(sg).execute(program)).lower()
                r = EvalResult(
                    question_id=sample['question_id'],
                    question=sample['question'],
                    expected_answer=sample['answer'],
                    predicted_answer=predicted,
                    correct=predicted == str(sample['answer']).lower(),
                    category=cat,
                    track='gt',
                    program_status='valid',
                    execution_error=None,
                )
            except ExecutionError as e:
                r = EvalResult(
                    question_id=sample['question_id'],
                    question=sample['question'],
                    expected_answer=sample['answer'],
                    predicted_answer=None,
                    correct=False,
                    category=cat,
                    track='gt',
                    program_status='valid',
                    execution_error=str(e),
                )

        results_new.append(r)
        ckpt_file.write(json.dumps(r.__dict__) + '\n')
        ckpt_file.flush()

        if r.correct:
            correct_so_far += 1

        if hasattr(bar, 'set_postfix'):
            done = len(done_ids) + len(results_new)
            bar.set_postfix(done=done, acc=f'{correct_so_far/len(results_new):.3f}')
        elif (i + 1) % 50 == 0:
            done = len(done_ids) + len(results_new)
            print(f'  {done}/{total} | acc={correct_so_far/len(results_new):.3f} | {time.time()-t0:.0f}s')

    ckpt_file.close()
    print(f'Done in {time.time()-t0:.1f}s')

    all_results = [EvalResult(**v) for v in done_ids.values()] + results_new
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   default='clevr_val', choices=list(DATASETS.keys()))
    parser.add_argument('--llm-model', default='llama3.2:3b')
    parser.add_argument('--limit',     type=int, default=1000)
    parser.add_argument('--output',    default='results')
    args = parser.parse_args()

    jsonl_path = DATASETS[args.dataset]
    if not Path(jsonl_path).exists():
        print(f'Dataset not found: {jsonl_path}')
        return

    llm_tag = args.llm_model.replace(':', '_').replace('.', '')
    tag = f'{args.dataset}_llm_{llm_tag}_n{args.limit}'

    print(f'Model: {args.llm_model}')
    print(f'Dataset: {args.dataset} ({args.limit} questions)')
    print(f'Output tag: {tag}')

    model = LLMTranslator(model=args.llm_model)

    ckpt = Path(args.output) / f'{tag}_checkpoint.jsonl'
    results = run_llm_gt_tok(jsonl_path, model, args.limit, str(ckpt))

    summary = compute_summary(results, track='gt', dataset=args.dataset)
    summary.print()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_q = out_dir / f'{tag}_per_question.jsonl'
    with open(per_q, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + '\n')

    summary_path = out_dir / f'{tag}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary.__dict__, f, indent=2)

    print(f'Saved: {per_q}')
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
