"""
    python -m eval.error_analysis \
        --gt    results/clevr_val_gt_n500_per_question.jsonl \
        --detected results/clevr_val_detected_n500_per_question.jsonl \
        --output results/error_analysis.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FailureBreakdown:
    total: int = 0

    # GT tok failures
    gt_nl2dsl_fail: int = 0 # program is None
    gt_execution_error: int = 0 # program valid but executor raised
    gt_wrong_answer: int = 0 # executed but answer wrong

    # Detected tok failures
    det_nl2dsl_fail: int = 0
    det_execution_error: int = 0
    det_wrong_answer: int = 0
    det_image_missing: int = 0

    # Attribution of detected-only errors
    # (correct in GT, wrong in detected → pure perception error)
    perception_errors: int = 0
    both_wrong: int = 0  # wrong in GT too → not a perception issue
    both_correct: int = 0

    by_category: Dict[str, Dict] = field(default_factory=dict)

    def print(self):
        print(f"\n=== Error Attribution ===")
        print(f"Total paired questions: {self.total}")
        print(f"\nGT tok failures:")
        print(f"  NL2DSL fail:      {self.gt_nl2dsl_fail:4d} ({100*self.gt_nl2dsl_fail/max(self.total,1):.1f}%)")
        print(f"  Execution error:  {self.gt_execution_error:4d} ({100*self.gt_execution_error/max(self.total,1):.1f}%)")
        print(f"  Wrong answer:     {self.gt_wrong_answer:4d} ({100*self.gt_wrong_answer/max(self.total,1):.1f}%)")
        print(f"\nDetected tok failures:")
        print(f"  NL2DSL fail:      {self.det_nl2dsl_fail:4d} ({100*self.det_nl2dsl_fail/max(self.total,1):.1f}%)")
        print(f"  Execution error:  {self.det_execution_error:4d} ({100*self.det_execution_error/max(self.total,1):.1f}%)")
        print(f"  Wrong answer:     {self.det_wrong_answer:4d} ({100*self.det_wrong_answer/max(self.total,1):.1f}%)")
        print(f"  Image missing:    {self.det_image_missing:4d} ({100*self.det_image_missing/max(self.total,1):.1f}%)")
        print(f"\nAttribution:")
        print(f"  Both correct:     {self.both_correct:4d} ({100*self.both_correct/max(self.total,1):.1f}%)")
        print(f"  Pure perception:  {self.perception_errors:4d} ({100*self.perception_errors/max(self.total,1):.1f}%)")
        print(f"  Both wrong:       {self.both_wrong:4d} ({100*self.both_wrong/max(self.total,1):.1f}%)")
        if self.by_category:
            print(f"\nPerception errors by category:")
            for cat, counts in sorted(self.by_category.items()):
                pct = 100 * counts['perception'] / max(counts['total'], 1)
                print(f"  {cat:10s}: {counts['perception']:3d}/{counts['total']:3d} ({pct:.1f}%)")


def load_results(path: str) -> Dict[str, dict]:
    """Load per-question JSONL, keyed by question_id."""
    results = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            results[r['question_id']] = r
    return results


def analyze(gt_path: str, detected_path: str) -> FailureBreakdown:
    gt = load_results(gt_path)
    det = load_results(detected_path)

    # Only analyze questions present in both
    common_ids = set(gt.keys()) & set(det.keys())

    b = FailureBreakdown(total=len(common_ids))
    category_counts = defaultdict(lambda: {'total': 0, 'perception': 0})

    for qid in common_ids:
        g = gt[qid]
        d = det[qid]
        cat = g.get('category', 'other')
        category_counts[cat]['total'] += 1

        # GT failures
        if g['program_status'] == 'failed':
            b.gt_nl2dsl_fail += 1
        elif g['execution_error']:
            b.gt_execution_error += 1
        elif not g['correct']:
            b.gt_wrong_answer += 1

        # Detected failures
        if d['program_status'] == 'failed':
            b.det_nl2dsl_fail += 1
        elif d['execution_error'] == 'image_not_found':
            b.det_image_missing += 1
        elif d['execution_error']:
            b.det_execution_error += 1
        elif not d['correct']:
            b.det_wrong_answer += 1

        # Attribution
        gt_correct = g['correct']
        det_correct = d['correct']

        if gt_correct and det_correct:
            b.both_correct += 1
        elif gt_correct and not det_correct:
            # GT works, detected fails → pure perception error
            b.perception_errors += 1
            category_counts[cat]['perception'] += 1
        else:
            b.both_wrong += 1

    b.by_category = dict(category_counts)
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='GT tok per_question.jsonl')
    parser.add_argument('--detected', required=True, help='Detected tok per_question.jsonl')
    parser.add_argument('--output', default='results/error_analysis.json')
    args = parser.parse_args()

    breakdown = analyze(args.gt, args.detected)
    breakdown.print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        data = breakdown.__dict__.copy()
        json.dump(data, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
