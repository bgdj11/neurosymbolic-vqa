from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class RelationConfig:
    
    # Horizontal relation thresholds
    # Objects are "left/right" if their centers differ by more than this (normalized [0,1])
    horizontal_threshold: float = 0.05  # 5% of image width
    horizontal_hysteresis: float = 0.02  # Hysteresis margin
    
    # Depth relation thresholds
    # Objects are "front/behind" if depth differs by more than this
    depth_threshold: float = 0.1  # 10% of depth range
    depth_hysteresis: float = 0.03
    
    # Minimum overlap for objects to be considered "at same position"
    position_overlap_iou: float = 0.1


class SpatialRelationComputer:
    
    def __init__(self, config: Optional[RelationConfig] = None):
        self.config = config or RelationConfig()
    
    def compute_all_relations(
        self,
        objects: List[Dict],
        depth_map: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = None,  # (H, W)
    ) -> Dict[str, List[List[int]]]:

        # Returns: Dictionary with relation lists: {'left': [[...], ...], 'right': [...], ...}
        n = len(objects)
        
        relations = {
            'left': [[] for _ in range(n)],
            'right': [[] for _ in range(n)],
            'front': [[] for _ in range(n)],
            'behind': [[] for _ in range(n)],
        }
        
        if n == 0:
            return relations
        
        # Extract centers and depths
        centers = []
        depths = []
        
        for obj in objects:
            bbox = obj.get('bbox', (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
            
            # Use depth_mean if available
            depth = obj.get('depth_mean', None)
            depths.append(depth)
        
        # Normalize centers if image_size provided
        if image_size:
            H, W = image_size
            centers = [(cx / W, cy / H) for cx, cy in centers]
        
        # Normalize depths to [0, 1] range
        valid_depths = [d for d in depths if d is not None]
        if valid_depths:
            depth_min = min(valid_depths)
            depth_max = max(valid_depths)
            depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
            depths_norm = [
                (d - depth_min) / depth_range if d is not None else None
                for d in depths
            ]
        else:
            depths_norm = [None] * n
        
        # Compute pairwise relations
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Horizontal relations (left/right)
                h_rel = self._compute_horizontal_relation(
                    centers[i], centers[j]
                )
                if h_rel == 'left':
                    relations['left'][i].append(j)
                elif h_rel == 'right':
                    relations['right'][i].append(j)
                
                # Depth relations (front/behind)
                d_rel = self._compute_depth_relation(
                    depths_norm[i], depths_norm[j]
                )
                if d_rel == 'front':
                    relations['front'][i].append(j)
                elif d_rel == 'behind':
                    relations['behind'][i].append(j)
        
        return relations
    
    def _compute_horizontal_relation(
        self,
        center_i: Tuple[float, float],
        center_j: Tuple[float, float],
    ) -> Optional[str]:

        cx_i, _ = center_i
        cx_j, _ = center_j
        
        diff = cx_j - cx_i  # Positive means j is to the right of i
        
        threshold = self.config.horizontal_threshold
        hysteresis = self.config.horizontal_hysteresis
        
        if diff > threshold + hysteresis:
            return 'right'  # j is to the right of i
        elif diff < -(threshold + hysteresis):
            return 'left'   # j is to the left of i
        else:
            return None     # Too close to determine
    
    def _compute_depth_relation(
        self,
        depth_i: Optional[float],
        depth_j: Optional[float],
    ) -> Optional[str]:

        if depth_i is None or depth_j is None:
            return None
        
        diff = depth_j - depth_i  # Positive means j is closer (in front)
        
        threshold = self.config.depth_threshold
        hysteresis = self.config.depth_hysteresis
        
        if diff > threshold + hysteresis:
            return 'front'   # j is in front of i
        elif diff < -(threshold + hysteresis):
            return 'behind'  # j is behind i
        else:
            return None      # Too similar depth
    
    def compute_overlap_relations(
        self,
        objects: List[Dict],
        iou_threshold: float = 0.1,
    ) -> Dict[str, List[List[int]]]:

        n = len(objects)
        
        relations = {
            'overlaps': [[] for _ in range(n)],
            'contains': [[] for _ in range(n)],
        }
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                bbox_i = objects[i].get('bbox', (0, 0, 0, 0))
                bbox_j = objects[j].get('bbox', (0, 0, 0, 0))
                
                iou = self._compute_iou(bbox_i, bbox_j)
                
                if iou > iou_threshold:
                    relations['overlaps'][i].append(j)
                
                # Check containment
                if self._is_contained(bbox_j, bbox_i):
                    relations['contains'][i].append(j)
        
        return relations
    
    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        # Compute Intersection over Union of two bounding boxes
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _is_contained(
        self,
        inner_bbox: Tuple[float, float, float, float],
        outer_bbox: Tuple[float, float, float, float],
        margin: float = 0.9,  # 90% of inner must be inside outer
    ) -> bool:
        # Check if inner_bbox is mostly contained in outer_bbox
        x1_in, y1_in, x2_in, y2_in = inner_bbox
        x1_out, y1_out, x2_out, y2_out = outer_bbox
        
        # Intersection
        xi1 = max(x1_in, x1_out)
        yi1 = max(y1_in, y1_out)
        xi2 = min(x2_in, x2_out)
        yi2 = min(y2_in, y2_out)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        inner_area = (x2_in - x1_in) * (y2_in - y1_in)
        
        if inner_area <= 0:
            return False
        
        return (inter_area / inner_area) >= margin


def compute_relations_for_detections(
    detections: List[Dict],
    depth_map: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    config: Optional[RelationConfig] = None,
) -> Dict[str, List[List[int]]]:

    #Returns:Dictionary of relations compatible with SceneGraph format.
    
    computer = SpatialRelationComputer(config)
    return computer.compute_all_relations(detections, depth_map, image_size)
