from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import json

from .relations import SpatialRelationComputer, RelationConfig


class SceneGraphBuilder:

    def __init__(
        self,
        detector=None,
        segmenter=None,
        depth_estimator=None,
        attribute_extractor=None,
        relation_config: Optional[RelationConfig] = None,
    ):
        self._detector = detector
        self._segmenter = segmenter
        self._depth_estimator = depth_estimator
        self._attribute_extractor = attribute_extractor
        self.relation_config = relation_config or RelationConfig()
        self.relation_computer = SpatialRelationComputer(self.relation_config)
        self._initialized = False
    
    def _lazy_init(self):
        if self._initialized:
            return
        
        if self._detector is None:
            from perception.detector import HybridDetector
            self._detector = HybridDetector()
            print("SceneGraphBuilder: Initialized HybridDetector")
        
        if self._segmenter is None:
            from perception.box_segmenter import BoxSegmenter
            self._segmenter = BoxSegmenter()
            print("SceneGraphBuilder: Initialized BoxSegmenter")
        
        if self._depth_estimator is None:
            from perception.depth_estimator import DepthEstimator
            self._depth_estimator = DepthEstimator()
            print("SceneGraphBuilder: Initialized DepthEstimator")
        
        if self._attribute_extractor is None:
            from perception.attribute_extractor import AttributeExtractor
            self._attribute_extractor = AttributeExtractor()
            print("SceneGraphBuilder: Initialized AttributeExtractor")
        
        self._initialized = True
    
    def build(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        use_segmentation: bool = True,
        use_depth: bool = True,
        use_heuristics: bool = True,
        return_intermediates: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict]]:

        self._lazy_init()
        
        intermediates = {}
        
        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            pil_image = Image.open(image_path).convert('RGB')
            np_image = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
            np_image = np.array(pil_image)
            image_path = None
        elif isinstance(image, np.ndarray):
            np_image = image
            pil_image = Image.fromarray(image)
            image_path = None
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        H, W = np_image.shape[:2]
        
        # Step 1: Detection
        if image_path:
            detections = self._detector.detect(image_path)
        else:
            detections = self._detector.detect(np_image)
        intermediates['detections'] = detections
        
        # Empty scene graph if no detections
        if len(detections) == 0:
            scene_graph = {
                'objects': [],
                'relationships': {'left': [], 'right': [], 'front': [], 'behind': []}
            }
            if return_intermediates:
                return scene_graph, intermediates
            return scene_graph
        
        # Step 2: Segmentation (optional)
        if use_segmentation:
            if image_path:
                segments = self._segmenter.segment(image_path, detections)
            else:
                segments = self._segmenter.segment(np_image, detections)
            intermediates['segments'] = segments
        else:
            segments = detections
        
        # Step 3: Depth estimation (optional)
        depth_map = None
        if use_depth:
            depth_map = self._depth_estimator.predict(np_image)
            intermediates['depth_map'] = depth_map
            
            from perception.depth_estimator import attach_depth_to_detections
            attach_depth_to_detections(segments, depth_map)
            
            from perception.postprocess import resolve_mask_overlaps_by_depth
            resolve_mask_overlaps_by_depth(segments, depth_map)
        
        # Step 4: Attribute extraction
        enriched = self._attribute_extractor.extract_from_detections(
            pil_image, segments, use_heuristics=use_heuristics
        )
        intermediates['enriched_detections'] = enriched
        
        # Step 5: Build objects list in CLEVR format
        objects = []
        for det in enriched:
            bbox = det.get('bbox', (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            depth_val = det.get('depth_mean', 0.0) or 0.0
            
            attrs = det.get('attributes', {})
            
            obj = {
                'color': attrs.get('color', {}).get('value'),
                'size': attrs.get('size', {}).get('value'),
                'shape': attrs.get('shape', {}).get('value'),
                'material': attrs.get('material', {}).get('value'),
                'pixel_coords': [cx, cy, depth_val],
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
            }
            objects.append(obj)
        
        # Step 6: Compute relations
        relationships = self.relation_computer.compute_all_relations(
            enriched, depth_map, image_size=(H, W)
        )
        
        # Build final scene graph
        scene_graph = {
            'objects': objects,
            'relationships': relationships,
        }
        
        if return_intermediates:
            return scene_graph, intermediates
        return scene_graph


def build_scene_graph(
    image: Union[str, Path, Image.Image, np.ndarray],
    **kwargs
) -> Dict[str, Any]:

    builder = SceneGraphBuilder()
    return builder.build(image, **kwargs)


def save_scene_graph(scene_graph: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(scene_graph, f, indent=2)


def load_scene_graph(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)
