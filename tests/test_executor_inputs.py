import pytest

from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph


@pytest.mark.parametrize('reference', [-1, 2, '0', True])
def test_executor_rejects_invalid_references(reference):
    scene = SceneGraph({'objects': [{'color': 'red'}]})
    program = [
        {'function': 'scene', 'inputs': []},
        {'function': 'count', 'inputs': [reference]},
    ]
    with pytest.raises(ExecutionError, match='Invalid input reference'):
        Executor(scene).execute(program)
