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


def equal_color(scene_graph: SceneGraph, objects1: List[int], objects2: List[int]) -> bool:
    if len(objects1) != 1 or len(objects2) != 1:
        raise ValueError(
            f"equal_color expects 1 object in each set, "
            f"got {len(objects1)} and {len(objects2)}"
        )
    color1 = scene_graph.get_attribute(objects1[0], 'color')
    color2 = scene_graph.get_attribute(objects2[0], 'color')
    return color1 == color2

def equal_shape(scene_graph: SceneGraph, objects1: List[int], objects2: List[int]) -> bool:
    if len(objects1) != 1 or len(objects2) != 1:
        raise ValueError(
            f"equal_shape expects 1 object in each set, "
            f"got {len(objects1)} and {len(objects2)}"
        )
    shape1 = scene_graph.get_attribute(objects1[0], 'shape')
    shape2 = scene_graph.get_attribute(objects2[0], 'shape')
    return shape1 == shape2

def equal_size(scene_graph: SceneGraph, objects1: List[int], objects2: List[int]) -> bool:
    if len(objects1) != 1 or len(objects2) != 1:
        raise ValueError(
            f"equal_size expects 1 object in each set, "
            f"got {len(objects1)} and {len(objects2)}"
        )
    size1 = scene_graph.get_attribute(objects1[0], 'size')
    size2 = scene_graph.get_attribute(objects2[0], 'size')
    return size1 == size2

def equal_material(scene_graph: SceneGraph, objects1: List[int], objects2: List[int]) -> bool:
    if len(objects1) != 1 or len(objects2) != 1:
        raise ValueError(
            f"equal_material expects 1 object in each set, "
            f"got {len(objects1)} and {len(objects2)}"
        )
    material1 = scene_graph.get_attribute(objects1[0], 'material')
    material2 = scene_graph.get_attribute(objects2[0], 'material')
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
    ref_color = scene_graph.get_attribute(objects[0], 'color')
    return [idx for idx in scene_graph.all_objects() if scene_graph.get_attribute(idx, 'color') == ref_color]

def same_shape(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_shape expects 1 object, got {len(objects)}: {objects}")
    ref_shape = scene_graph.get_attribute(objects[0], 'shape')
    return [idx for idx in scene_graph.all_objects() if scene_graph.get_attribute(idx, 'shape') == ref_shape]

def same_size(scene_graph: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_size expects 1 object, got {len(objects)}: {objects}")
    ref_size = scene_graph.get_attribute(objects[0], 'size')
    return [idx for idx in scene_graph.all_objects() if scene_graph.get_attribute(idx, 'size') == ref_size]

def same_material(sg: SceneGraph, objects: List[int]) -> List[int]:
    if len(objects) != 1:
        raise ValueError(f"same_material expects 1 object, got {len(objects)}: {objects}")
    
    ref_material = sg.get_attribute(objects[0], 'material')
    return [idx for idx in sg.all_objects() if sg.get_attribute(idx, 'material') == ref_material]

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
