import json
from pathlib import Path

import pytest

from dsl.scene_graph import SceneGraph
from dsl.executor import Executor, ExecutionError

CLEVR_VAL = Path('datasets/converted/clevr/clevr_val.jsonl')
FAST_LIMIT = 200
FULL_LIMIT = 10_000


def _load_clevr_samples(n: int):
    if not CLEVR_VAL.exists():
        return []
    samples = []
    with open(CLEVR_VAL, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))
    return samples


_FAST_SAMPLES = _load_clevr_samples(FAST_LIMIT)

# Fast parametrized suite (runs by default)

@pytest.mark.skipif(not CLEVR_VAL.exists(), reason="CLEVR val JSONL not found")
@pytest.mark.parametrize("sample", _FAST_SAMPLES, ids=[s['question_id'] for s in _FAST_SAMPLES])
def test_executor_on_clevr(sample):
    sg = SceneGraph(sample['scene_graph'])
    result = Executor(sg).execute(sample['program'])
    assert str(result).lower() == str(sample['answer']).lower(), (
        f"Question: {sample['question']}\n"
        f"Expected: {sample['answer']!r}, Got: {result!r}"
    )


# Unit tests for executor behaviour (no dataset needed)

def _simple_scene():
    """Two objects: a red cube and a blue sphere."""
    return SceneGraph({
        'objects': [
            {'color': 'red', 'shape': 'cube', 'size': 'small', 'material': 'rubber'},
            {'color': 'blue', 'shape': 'sphere', 'size': 'large', 'material': 'metal'},
        ],
        'relationships': {
            'left':   [[], [0]],
            'right':  [[1], []],
            'front':  [[], []],
            'behind': [[], []],
        }
    })


def test_count_all_objects():
    sg = _simple_scene()
    result = Executor(sg).execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'count', 'inputs': [0], 'value_inputs': [], 'side_inputs': []},
    ])
    assert result == 2


def test_filter_color_red():
    sg = _simple_scene()
    result = Executor(sg).execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'filter_color', 'inputs': [0], 'value_inputs': ['red'], 'side_inputs': []},
        {'function': 'count', 'inputs': [1], 'value_inputs': [], 'side_inputs': []},
    ])
    assert result == 1


def test_exist_returns_yes_no():
    sg = _simple_scene()
    yes_result = Executor(sg).execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'filter_color', 'inputs': [0], 'value_inputs': ['red'], 'side_inputs': []},
        {'function': 'exist', 'inputs': [1], 'value_inputs': [], 'side_inputs': []},
    ])
    assert yes_result == 'yes'

    no_result = Executor(sg).execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'filter_color', 'inputs': [0], 'value_inputs': ['green'], 'side_inputs': []},
        {'function': 'exist', 'inputs': [1], 'value_inputs': [], 'side_inputs': []},
    ])
    assert no_result == 'no'


def test_query_color():
    sg = _simple_scene()
    result = Executor(sg).execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'filter_shape', 'inputs': [0], 'value_inputs': ['cube'], 'side_inputs': []},
        {'function': 'unique', 'inputs': [1], 'value_inputs': [], 'side_inputs': []},
        {'function': 'query_color', 'inputs': [2], 'value_inputs': [], 'side_inputs': []},
    ])
    assert result == 'red'


def test_empty_program_raises():
    sg = _simple_scene()
    with pytest.raises(ExecutionError):
        Executor(sg).execute([])


def test_unknown_operator_raises():
    sg = _simple_scene()
    with pytest.raises(ExecutionError):
        Executor(sg).execute([
            {'function': 'nonexistent_op', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        ])


def test_get_execution_trace():
    sg = _simple_scene()
    ex = Executor(sg)
    ex.execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'count', 'inputs': [0], 'value_inputs': [], 'side_inputs': []},
    ])
    trace = ex.get_execution_trace()
    assert len(trace) == 2
    assert trace[0]['function'] == 'scene'
    assert trace[1]['function'] == 'count'
    assert trace[1]['result'] == 2


# Full CLEVR val run (opt-in)

@pytest.mark.clevr_full
@pytest.mark.skipif(not CLEVR_VAL.exists(), reason="CLEVR val JSONL not found")
def test_executor_full_clevr_val():
    """Run executor on up to 10 000 CLEVR val samples. Opt-in marker."""
    samples = _load_clevr_samples(FULL_LIMIT)
    success = mismatches = errors = 0

    for sample in samples:
        try:
            sg = SceneGraph(sample['scene_graph'])
            result = Executor(sg).execute(sample['program'])
            if str(result).lower() == str(sample['answer']).lower():
                success += 1
            else:
                mismatches += 1
        except Exception:
            errors += 1

    total = success + mismatches + errors
    print(f"\nFull CLEVR val: {success}/{total} correct, {mismatches} mismatches, {errors} errors")
    assert errors == 0, f"{errors} execution errors on CLEVR val programs"
    assert mismatches == 0, f"{mismatches} answer mismatches on CLEVR val programs"
