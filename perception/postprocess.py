import numpy as np
from typing import List, Dict, Any


def resolve_mask_overlaps_by_depth(detections: List[Dict[str, Any]], depth_map: np.ndarray) -> None:
    H, W = depth_map.shape[:2]

    def depth_key(det: Dict[str, Any]) -> float:
        dm = det.get("depth_mean", None)
        if dm is None:
            return float("-inf")
        return dm
    
    order = sorted(range(len(detections)),
        key=lambda i: depth_key(detections[i]),
        reverse=True,
    )

    occupied = np.zeros((H, W), dtype=bool)

    for idx in order:
        det = detections[idx]
        mask = det.get("mask", None)
        if mask is None:
            continue
        if mask.shape != (H, W):
            continue

        m = mask.astype(bool)
        new_mask = np.logical_and(m, np.logical_not(occupied))

        if not new_mask.any():
            det["mask"] = None
            det["mask_source"] = det.get("mask_source", "") + "|pruned_by_depth"
            continue

        det["mask"] = new_mask
        occupied |= new_mask
