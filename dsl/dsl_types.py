from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class DataType(Enum):
    OBJECT_SET = "ObjectSet"
    BOOL = "Bool"
    INTEGER = "Integer"
    ATTRIBUTE = "Attribute"

@dataclass
class OperatorSignature:
    name: str
    inputs: List[DataType]
    value_inputs: List[str]
    side_inputs: Optional[List[str]]
    output: DataType
    description: str

OPERATOR_SIGNATURES: Dict[str, OperatorSignature] = {
    'scene': OperatorSignature(
        name='scene',
        inputs=[],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Return all objects in the scene'
    ),
    'filter_color': OperatorSignature(
        name='filter_color',
        inputs=[DataType.OBJECT_SET],
        value_inputs=['str'],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Filter objects by color attribute'
    ),
    'filter_shape': OperatorSignature(
        name='filter_shape',
        inputs=[DataType.OBJECT_SET],
        value_inputs=['str'],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Filter objects by shape attribute'
    ),
    'filter_size': OperatorSignature(
        name='filter_size',
        inputs=[DataType.OBJECT_SET],
        value_inputs=['str'],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Filter objects by size attribute'
    ),
    'filter_material': OperatorSignature(
        name='filter_material',
        inputs=[DataType.OBJECT_SET],
        value_inputs=['str'],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Filter objects by material attribute'
    ),
    'relate': OperatorSignature(
        name='relate',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=['str'],
        output=DataType.OBJECT_SET,
        description='Find all objects in given spatial relation to input objects'
    ),
    'intersect': OperatorSignature(
        name='intersect',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Return intersection of two object sets'
    ),
    'union': OperatorSignature(
        name='union',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Return union of two object sets'
    ),
    'unique': OperatorSignature(
        name='unique',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Assert that input contains exactly one object and return it'
    ),
    'query_color': OperatorSignature(
        name='query_color',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.ATTRIBUTE,
        description='Return color of the (unique) object'
    ),
    'query_shape': OperatorSignature(
        name='query_shape',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.ATTRIBUTE,
        description='Return shape of the (unique) object'
    ),
    'query_size': OperatorSignature(
        name='query_size',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.ATTRIBUTE,
        description='Return size of the (unique) object'
    ),
    'query_material': OperatorSignature(
        name='query_material',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.ATTRIBUTE,
        description='Return material of the (unique) object'
    ),
    'count': OperatorSignature(
        name='count',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.INTEGER,
        description='Return number of objects in the set'
    ),
    'exist': OperatorSignature(
        name='exist',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Return true if set is non-empty, false otherwise'
    ),
    'equal_color': OperatorSignature(
        name='equal_color',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if two objects have the same color'
    ),
    'equal_shape': OperatorSignature(
        name='equal_shape',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if two objects have the same shape'
    ),
    'equal_size': OperatorSignature(
        name='equal_size',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if two objects have the same size'
    ),
    'equal_material': OperatorSignature(
        name='equal_material',
        inputs=[DataType.OBJECT_SET, DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if two objects have the same material'
    ),
    'equal_integer': OperatorSignature(
        name='equal_integer',
        inputs=[DataType.INTEGER, DataType.INTEGER],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if two integers are equal'
    ),
    'less_than': OperatorSignature(
        name='less_than',
        inputs=[DataType.INTEGER, DataType.INTEGER],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if first integer is less than second'
    ),
    
    'greater_than': OperatorSignature(
        name='greater_than',
        inputs=[DataType.INTEGER, DataType.INTEGER],
        value_inputs=[],
        side_inputs=None,
        output=DataType.BOOL,
        description='Check if first integer is greater than second'
    ),
    'same_color': OperatorSignature(
        name='same_color',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Find all objects with same color as input object'
    ),
    'same_shape': OperatorSignature(
        name='same_shape',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Find all objects with same shape as input object'
    ),
    'same_size': OperatorSignature(
        name='same_size',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Find all objects with same size as input object'
    ),
    'same_material': OperatorSignature(
        name='same_material',
        inputs=[DataType.OBJECT_SET],
        value_inputs=[],
        side_inputs=None,
        output=DataType.OBJECT_SET,
        description='Find all objects with same material as input object'
    ),
}

VALID_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
VALID_SHAPES = ['cube', 'sphere', 'cylinder']
VALID_SIZES = ['small', 'large']
VALID_MATERIALS = ['rubber', 'metal']
VALID_RELATIONS = ['left', 'right', 'front', 'behind']



def get_operator_signature(operator_name: str) -> Optional[OperatorSignature]:
    return OPERATOR_SIGNATURES.get(operator_name)


def is_valid_operator(operator_name: str) -> bool:
    return operator_name in OPERATOR_SIGNATURES


def get_all_operator_names() -> List[str]:
    return list(OPERATOR_SIGNATURES.keys())


def get_operators_by_output_type(output_type: DataType) -> List[str]:
    return [
        name for name, sig in OPERATOR_SIGNATURES.items()
        if sig.output == output_type
    ]

def print_operator_summary():
    print()
    print("DSL OPERATOR SUMMARY")
    print()

    categories = {
        'Scene': ['scene'],
        'Filter': [n for n in OPERATOR_SIGNATURES if n.startswith('filter_')],
        'Relate': ['relate'],
        'Set Operations': ['intersect', 'union', 'unique'],
        'Query': [n for n in OPERATOR_SIGNATURES if n.startswith('query_')],
        'Count/Exist': ['count', 'exist'],
        'Comparison': [n for n in OPERATOR_SIGNATURES if n.startswith('equal_') or n in ['less_than', 'greater_than']],
        'Same': [n for n in OPERATOR_SIGNATURES if n.startswith('same_')],
    }
    
    for category, ops in categories.items():
        print(f"{category} ({len(ops)} operators):")
        for op_name in ops:
            sig = OPERATOR_SIGNATURES[op_name]
            print(f"    - {op_name}: {sig.description}")
        print()
    
    print(f"TOTAL: {len(OPERATOR_SIGNATURES)} operators")
    print()


if __name__ == '__main__':
    print_operator_summary()
