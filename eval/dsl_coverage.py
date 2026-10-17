"""Estimate DSL coverage with lexical rules; this is a heuristic classification."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_SUPERLATIVE_TOKENS = {
    'furthest', 'farthest', 'closest', 'nearest',
    'leftmost', 'rightmost', 'topmost', 'backmost',
    'tallest', 'shortest',
}

_SUPERLATIVE_PHRASES = [
    r'\bfurthest\b', r'\bfarthest\b', r'\bclosest\b', r'\bnearest\b',
    r'\bleftmost\b', r'\brightmost\b',
    r'\bmost common\b', r'\bmost frequent\b', r'\bdominant\b',
    r'\bmost .{1,20}\b(color|shape|material|size)',
    r'\bfewest\b', r'\bmost\b',
]

_COUNTERFACTUAL_PHRASES = [
    r'\bif .{1,30} (were|was|are) (added|removed|taken away|included)\b',
    r'\bwould (there|it|they) be\b',
    r'\bhow many times\b',
    r'\bhow many .{1,30} (more|fewer|less) than\b',
    r'\bhow many more .{1,40} than\b',
    r'\bhow much\b',
]

_BETWEEN_PHRASES = [
    r'\bin (the )?middle\b',
    r'\bbetween .{1,30} and\b',
    r'\bbetween the\b',
    r'\bsandwiched\b',
]

_OCCLUSION_PHRASES = [
    r'\bhiding\b', r'\bblocked\b', r'\bhidden\b', r'\bobscured\b',
    r'\bbehind .{0,20} (and|or) .{0,20}(visible|seen|showing)\b',
]

_COMPARATIVE_SELECT = [
    r'\bwhich (is|one is|are) (larger|smaller|bigger|closer|further|taller|shorter|more|fewer)\b',
    r'\bwhich .{1,20} (more|fewer)\b',
]

_FREQUENCY_PHRASES = [
    r'\bmost (often|commonly|frequently|represented)\b',
    r'\bappears? more\b',
    r'\boccurs? more\b',
]

_ALL_PATTERNS = (
    _SUPERLATIVE_PHRASES + _COUNTERFACTUAL_PHRASES +
    _BETWEEN_PHRASES + _OCCLUSION_PHRASES +
    _COMPARATIVE_SELECT + _FREQUENCY_PHRASES
)

_COMPILED = [(re.compile(p, re.IGNORECASE), p) for p in _ALL_PATTERNS]


def classify_dsl_coverage(question: str) -> tuple[bool, str]:
    """
    Vraca (expressible, reason).
    expressible=True ako DSL MOZE da izrazi pitanje,
    False ako SIGURNO ne moze (konzervativna klasifikacija).
    """
    q = question.lower().strip()
    for pattern, pat_str in _COMPILED:
        if pattern.search(q):
            if 'furthest' in pat_str or 'farthest' in pat_str or 'leftmost' in pat_str or 'most common' in pat_str or 'fewest' in pat_str or 'most' in pat_str:
                label = 'superlative'
            elif 'if' in pat_str or 'would' in pat_str or 'times' in pat_str or 'more than' in pat_str:
                label = 'counterfactual'
            elif 'middle' in pat_str or 'between' in pat_str:
                label = 'spatial_between'
            elif 'hiding' in pat_str or 'blocked' in pat_str:
                label = 'occlusion'
            elif 'which' in pat_str:
                label = 'comparative_select'
            else:
                label = 'frequency'
            return False, label
    return True, 'expressible'


def analyze_coverage_from_jsonl(jsonl_path: str, n: int = None):
    """Samo coverage analiza na JSONL datasetu (bez eval rezultata)."""
    samples = []
    with open(jsonl_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            samples.append(json.loads(line))

    covered = []
    uncovered = defaultdict(list)

    for s in samples:
        exp, reason = classify_dsl_coverage(s['question'])
        if exp:
            covered.append(s)
        else:
            uncovered[reason].append(s)

    total = len(samples)
    n_covered = len(covered)
    n_uncovered = total - n_covered

    print(f'\n{"="*56}')
    print(f'DSL COVERAGE ANALYSIS — {total} pitanja')
    print(f'{"="*56}')
    print(f'\n[DSL COVERAGE]')
    print(f'  Expressible:    {n_covered}/{total} = {n_covered/total:.1%}')
    print(f'  Not expressible:{n_uncovered}/{total} = {n_uncovered/total:.1%}')
    print(f'\n[RAZLOZI NEMOGUCNOSTI (uncovered)]')
    for reason, items in sorted(uncovered.items(), key=lambda x: -len(x[1])):
        pct = len(items) / total
        print(f'  {reason:<22} {len(items):>4}  ({pct:.1%})')
        for s in items[:2]:
            print(f'    e.g. "{s["question"][:75]}"')

    return covered, dict(uncovered)


def analyze_eval_results(results_jsonl: str, coverage_jsonl: str = None):
    """
    Ucitava per-question eval rezultate i dodaje DSL coverage klasifikaciju.
    results_jsonl: per_question.jsonl iz run_eval.py
    coverage_jsonl: opcioni poseban JSONL za coverage (ako eval nema sve samples)
    """
    results = []
    with open(results_jsonl, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            results.append(r)

    total = len(results)
    if total == 0:
        print('Nema rezultata.')
        return

    covered_results = []
    uncovered_results = []
    uncovered_reasons = Counter()

    for r in results:
        exp, reason = classify_dsl_coverage(r['question'])
        r['dsl_expressible'] = exp
        r['dsl_reason'] = reason
        if exp:
            covered_results.append(r)
        else:
            uncovered_results.append(r)
            uncovered_reasons[reason] += 1

    n_cov = len(covered_results)
    n_uncov = len(uncovered_results)

    def acc(lst):
        return sum(r['correct'] for r in lst) / len(lst) if lst else 0

    def exec_err_rate(lst):
        return sum(1 for r in lst if r.get('execution_error')) / len(lst) if lst else 0

    print(f'\n{"="*60}')
    print(f'CLEVR-HUMANS EVAL — SPLIT PO DSL COVERAGE')
    print(f'{"="*60}')

    print(f'\n[1] DSL COVERAGE')
    print(f'  Expressible:     {n_cov}/{total} = {n_cov/total:.1%}')
    print(f'  Not expressible: {n_uncov}/{total} = {n_uncov/total:.1%}')

    print(f'\n[2] ACCURACY ON COVERED SUBSET ({n_cov} pitanja)')
    cov_acc = acc(covered_results)
    print(f'  Overall:         {sum(r["correct"] for r in covered_results)}/{n_cov} = {cov_acc:.1%}')
    print(f'  Exec error rate: {exec_err_rate(covered_results):.1%}')
    by_cat = defaultdict(list)
    for r in covered_results:
        by_cat[r.get('category', 'other')].append(r['correct'])
    print(f'  By category:')
    for cat, vals in sorted(by_cat.items()):
        print(f'    {cat:<12} {sum(vals):>4}/{len(vals):<4} = {sum(vals)/len(vals):.1%}')

    print(f'\n[3] FAILURE ANALYSIS ON UNCOVERED SUBSET ({n_uncov} pitanja)')
    print(f'  Accuracy here:   {acc(uncovered_results):.1%}  (lower bound, DSL cant express)')
    print(f'  Razlozi:')
    for reason, cnt in uncovered_reasons.most_common():
        pct = cnt / n_uncov if n_uncov else 0
        print(f'    {reason:<22} {cnt:>4}  ({pct:.1%})')

    print(f'\n  Primeri pitanja koje DSL ne moze izraziti:')
    by_reason = defaultdict(list)
    for r in uncovered_results:
        by_reason[r['dsl_reason']].append(r)
    for reason, items in list(by_reason.items())[:3]:
        print(f'\n  [{reason}]')
        for r in items[:3]:
            print(f'    Q: {r["question"]}')
            print(f'    A: {r["expected_answer"]}  got: {r.get("predicted_answer","?")}  correct: {r["correct"]}')

    print(f'\n{"="*60}')
    print(f'UKUPNO (za referencu): {acc(results):.1%}  na svim {total} pitanjima')
    print(f'REALISTICALLY:         {cov_acc:.1%}  na DSL-expressible podskupu')
    print(f'{"="*60}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_results',   help='per_question.jsonl iz run_eval.py')
    parser.add_argument('--jsonl',          help='CLEVR-Humans JSONL za coverage-only analizu')
    parser.add_argument('--coverage_only',  action='store_true')
    parser.add_argument('--n',              type=int, default=None)
    args = parser.parse_args()

    if args.coverage_only or (args.jsonl and not args.eval_results):
        jsonl = args.jsonl or 'datasets/converted/clevr_humans/clevr_humans_eval.jsonl'
        analyze_coverage_from_jsonl(jsonl, args.n)
    elif args.eval_results:
        analyze_eval_results(args.eval_results)
    else:
        analyze_coverage_from_jsonl(
            'datasets/converted/clevr_humans/clevr_humans_eval.jsonl', args.n
        )


if __name__ == '__main__':
    main()
