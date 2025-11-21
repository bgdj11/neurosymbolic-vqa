from typing import List
from dsl.scene_graph import SceneGraph

def scene(scene_graph: SceneGraph) -> List[int]:
    return scene_graph.all_objects()


def filter_color(scene_graph: SceneGraph, objects: List[int], color: str) -> List[int]:
    return [idx for idx in objects if scene_graph.get_attribute(idx, 'color') == color]

def filter_shape(scene_graph: SceneGraph, objects: List[int], shape: str) -> List[int]:
    return [idx for idx in objects if scene_graph.get_attribute(idx, 'shape') == shape]

def filter_size(scene_graph: SceneGraph, objects: List[int], size: str) -> List[int]:
    return [idx for idx in objects if scene_graph.get_attribute(idx, 'size') == size]

def filter_material(scene_graph: SceneGraph, objects: List[int], material: str) -> List[int]:
    return [idx for idx in objects if scene_graph.get_attribute(idx, 'material') == material]


def relate(scene_graph: SceneGraph, objects: List[int], relation: str) -> List[int]:
    result = set() #unique objects
    for obj_idx in objects:
        related = scene_graph.get_related(obj_idx, relation)
        result.update(related)
    return list(result)


def intersect(scene_graph: SceneGraph, set1: List[int], set2: List[int]) -> List[int]:
    return list(set(set1) & set(set2))

def union(scene_graph: SceneGraph, set1: List[int], set2: List[int]) -> List[int]:
    return list(set(set1) | set(set2))


def unique(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(
            f"unique() expects exactly 1 object, got {len(objects)}: {objects}"
        )
    return objects


def query_color(scene_graph: SceneGraph, objects: List[int]) -> str:
    if len(objects) != 1:
        raise ValueError(f"query_color expects 1 object, got {len(objects)}: {objects}")
    return scene_graph.get_attribute(objects[0], 'color')

def query_shape(scene_graph: SceneGraph, objects: List[int]) -> str:
    if len(objects) != 1:
        raise ValueError(f"query_shape expects 1 object, got {len(objects)}: {objects}")
    return scene_graph.get_attribute(objects[0], 'shape')

def query_size(scene_graph: SceneGraph, objects: List[int]) -> str:
    if len(objects) != 1:
        raise ValueError(f"query_size expects 1 object, got {len(objects)}: {objects}")
    return scene_graph.get_attribute(objects[0], 'size')

def query_material(scene_graph: SceneGraph, objects: List[int]) -> str:
    if len(objects) != 1:
        raise ValueError(f"query_material expects 1 object, got {len(objects)}: {objects}")
    return scene_graph.get_attribute(objects[0], 'material')


def count(scene_graph: SceneGraph, objects: List[int]) -> int:
    return len(objects)


def exist(scene_graph: SceneGraph, objects: List[int]) -> bool:
    return len(objects) > 0


def equal_color(scene_graph: SceneGraph, color1: str, color2: str) -> bool:
    return color1 == color2

def equal_shape(scene_graph: SceneGraph, shape1: str, shape2: str) -> bool:
    return shape1 == shape2

def equal_size(scene_graph: SceneGraph, size1: str, size2: str) -> bool:
    return size1 == size2

def equal_material(scene_graph: SceneGraph, material1: str, material2: str) -> bool:
    return material1 == material2


def equal_integer(scene_graph: SceneGraph, int1: int, int2: int) -> bool:
    return int1 == int2

def less_than(scene_graph: SceneGraph, int1: int, int2: int) -> bool:
    return int1 < int2

def greater_than(scene_graph: SceneGraph, int1: int, int2: int) -> bool:
    return int1 > int2


def same_color(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_color expects 1 object, got {len(objects)}: {objects}")
    ref_obj = objects[0]
    ref_color = scene_graph.get_attribute(ref_obj, 'color')
    return [idx for idx in scene_graph.all_objects() 
            if idx != ref_obj and scene_graph.get_attribute(idx, 'color') == ref_color]

def same_shape(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_shape expects 1 object, got {len(objects)}: {objects}")
    ref_obj = objects[0]
    ref_shape = scene_graph.get_attribute(ref_obj, 'shape')
    return [idx for idx in scene_graph.all_objects() 
            if idx != ref_obj and scene_graph.get_attribute(idx, 'shape') == ref_shape]

def same_size(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_size expects 1 object, got {len(objects)}: {objects}")
    ref_obj = objects[0]
    ref_size = scene_graph.get_attribute(ref_obj, 'size')
    return [idx for idx in scene_graph.all_objects() 
            if idx != ref_obj and scene_graph.get_attribute(idx, 'size') == ref_size]

def same_material(sg: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_material expects 1 object, got {len(objects)}: {objects}")
    ref_obj = objects[0]
    ref_material = sg.get_attribute(ref_obj, 'material')
    return [idx for idx in sg.all_objects() 
            if idx != ref_obj and sg.get_attribute(idx, 'material') == ref_material]

OPERATORS = {
    'scene': scene,
    'filter_color': filter_color,
    'filter_shape': filter_shape,
    'filter_size': filter_size,
    'filter_material': filter_material,
    'relate': relate,
    'intersect': intersect,
    'union': union,
    'unique': unique,
    'query_color': query_color,
    'query_shape': query_shape,
    'query_size': query_size,
    'query_material': query_material,
    'count': count,
    'exist': exist,
    'equal_color': equal_color,
    'equal_shape': equal_shape,
    'equal_size': equal_size,
    'equal_material': equal_material,
    'equal_integer': equal_integer,
    'less_than': less_than,
    'greater_than': greater_than,
    'same_color': same_color,
    'same_shape': same_shape,
    'same_size': same_size,
    'same_material': same_material,
}


def get_operator(name: str):
    if name not in OPERATORS:
        raise KeyError(
            f"Unknown operator: '{name}'. "
            f"Available: {list(OPERATORS.keys())}"
        )
    return OPERATORS[name]
