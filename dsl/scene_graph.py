from typing import List, Dict, Any, Optional

class SceneGraph:
    def __init__(self, scene_data: Dict[str, Any]):
        self.objects = scene_data.get('objects', [])
        self.relationships = scene_data.get('relationships', {})
        self.num_objects = len(self.objects)
           
    def get_object(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_objects:
            raise IndexError(f"Object index {idx} out of range [0, {self.num_objects})")
        return self.objects[idx]
    
    def get_attribute(self, idx: int, attr: str) -> str:
        obj = self.get_object(idx)
        
        if attr not in obj:
            raise KeyError(
                f"Attribute '{attr}' not found in object {idx}. "
                f"Available: {list(obj.keys())}"
            )
        
        return obj[attr]
    
    def get_related(self, idx: int, relation: str) -> List[int]:
        if idx < 0 or idx >= self.num_objects:
            raise IndexError(f"Object index {idx} out of range [0, {self.num_objects})")
        
        if relation not in self.relationships:
            raise KeyError(
                f"Relation '{relation}' not found. "
                f"Available: {list(self.relationships.keys())}"
            )
        
        return self.relationships[relation][idx]
    
    def all_objects(self) -> List[int]:
        return list(range(self.num_objects))
    
    def has_objects(self) -> bool:
        return self.num_objects > 0
    
    def is_valid_index(self, idx: int) -> bool:
        return 0 <= idx < self.num_objects
    
    def get_object_description(self, idx: int) -> str:
        if idx < 0 or idx >= self.num_objects:
            raise IndexError(f"Object index {idx} out of range [0, {self.num_objects})")
        obj = self.get_object(idx)
        return f"{obj.get('size', '?')} {obj.get('color', '?')} {obj.get('material', '?')} {obj.get('shape', '?')}"
    
    def __len__(self) -> int:
        return self.num_objects
    
    def __repr__(self) -> str:
        return f"SceneGraph(objects={self.num_objects})"
    
    def __str__(self) -> str:
        lines = [f"SceneGraph with {self.num_objects} objects:"]
        for i in range(min(5, self.num_objects)):
            lines.append(f"  [{i}] {self.get_object_description(i)}")
        if self.num_objects > 5:
            lines.append(f"  ... and {self.num_objects - 5} more")
        return "\n".join(lines)

def create_scene_graph(scene_data: Dict[str, Any]) -> SceneGraph:
    return SceneGraph(scene_data)

def validate_scene_data(scene_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    try:
        SceneGraph(scene_data)
        return True, None
    except Exception as e:
        return False, str(e)


if __name__ == '__main__':

    # TEST example
    example_scene = {
        'objects': [
            {'color': 'red', 'shape': 'cube', 'size': 'small', 'material': 'metal'},
            {'color': 'blue', 'shape': 'sphere', 'size': 'large', 'material': 'rubber'},
            {'color': 'green', 'shape': 'cylinder', 'size': 'small', 'material': 'metal'},
        ],
        'relationships': {
            'left': [[1], [], [0, 1]],
            'right': [[], [0, 2], []],
            'front': [[2], [0], []],
            'behind': [[], [], [0]],
        }
    }
    
    scene = SceneGraph(example_scene)
    
    print(scene)
    print()
    print("Testing SceneGraph API:")
    print(f"  Total objects: {len(scene)}")
    print(f"  All object indices: {scene.all_objects()}")
    print()
    
    print("Object 0:")
    print(f"  Color: {scene.get_attribute(0, 'color')}")
    print(f"  Shape: {scene.get_attribute(0, 'shape')}")
    print(f"  Description: {scene.get_object_description(0)}")
    print(f"  Left of object 0: {scene.get_related(0, 'left')}")
    print(f"  Right of object 0: {scene.get_related(0, 'right')}")
    print()
