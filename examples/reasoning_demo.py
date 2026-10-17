"""Run a complete symbolic trace without models or datasets."""

from dsl.executor import Executor
from dsl.scene_graph import SceneGraph
from dsl.validator import DSLValidator
from nl2dsl.prepare_data import linear_to_program


def main():
    scene = SceneGraph({
        'objects': [
            {'color': 'cyan', 'shape': 'sphere', 'size': 'small', 'material': 'rubber'},
            {'color': 'blue', 'shape': 'cube', 'size': 'large', 'material': 'metal'},
        ],
        'relationships': {
            'left': [[], []], 'right': [[], []],
            'behind': [[1], []], 'front': [[], [0]],
        },
    })
    question = 'What color is the cube behind the cyan sphere?'
    linear = (
        'scene | filter_color cyan | filter_shape sphere | unique | '
        'relate behind | filter_shape cube | unique | query_color'
    )
    program = linear_to_program(linear)
    validation = DSLValidator().validate(program)
    if not validation.is_valid:
        raise ValueError(validation.errors)
    print(question)
    answer = Executor(scene, verbose=True).execute(program)
    assert answer == 'blue'


if __name__ == '__main__':
    main()
