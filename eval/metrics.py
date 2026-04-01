from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalResult:
    question_id: str
    question: str
    expected_answer: str
    predicted_answer: Optional[str]
    correct: bool
    category: str # count | exist | query | compare | other
    track: str # 'gt' | 'detected'
    program_status: str  # 'valid' | 'failed'
    execution_error: Optional[str]


@dataclass
class EvalSummary:
    track: str
    dataset: str
    total: int
    correct: int
    accuracy: float
    accuracy_by_category: Dict[str, float]
    valid_program_rate: float
    execution_error_rate: float
    failed_program_count: int

    def print(self):
        print(f'\n=== {self.dataset} | {self.track} tok ===')
        print(f'Accuracy:          {self.correct}/{self.total} = {self.accuracy:.1%}')
        print(f'Valid program rate: {self.valid_program_rate:.1%}')
        print(f'Execution errors:  {self.execution_error_rate:.1%}')
        print(f'Failed NL2DSL:     {self.failed_program_count}')
        print('By category:')
        for cat, acc in sorted(self.accuracy_by_category.items()):
            print(f'  {cat:<10} {acc:.1%}')


def categorize_question(question: str) -> str:
    q = question.lower()
    if q.startswith('how many'):
        return 'count'
    if q.startswith('is there') or q.startswith('are there'):
        return 'exist'
    if q.startswith('what') or q.startswith('which'):
        return 'query'
    if 'same' in q or 'more' in q or 'fewer' in q or 'equal' in q:
        return 'compare'
    return 'other'


def compute_summary(results: List[EvalResult], track: str, dataset: str) -> EvalSummary:
    total = len(results)
    correct = sum(r.correct for r in results)
    valid = sum(r.program_status == 'valid' for r in results)
    exec_errors = sum(r.execution_error is not None for r in results)
    failed = sum(r.program_status == 'failed' for r in results)

    by_cat: Dict[str, List[bool]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.correct)
    acc_by_cat = {cat: sum(v) / len(v) for cat, v in by_cat.items()}

    return EvalSummary(
        track=track,
        dataset=dataset,
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        accuracy_by_category=acc_by_cat,
        valid_program_rate=valid / total if total else 0.0,
        execution_error_rate=exec_errors / total if total else 0.0,
        failed_program_count=failed,
    )
