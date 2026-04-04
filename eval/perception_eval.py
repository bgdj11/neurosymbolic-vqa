"""
    python -m eval.perception_eval \
        --jsonl datasets/converted/clevr/clevr_val.jsonl \
        --images datasets/clevr/images/val \
        --limit 200 \
        --output results/perception_eval.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

RELATION_TYPES = ['left', 'right', 'front', 'behind']



def match_objects(gt_objects, det_objects) -> List[Tuple[int, int]]:

    if not gt_objects or not det_objects:
        return []

    pairs = []
    used_det = set()

    for gi, gt_obj in enumerate(gt_objects):
        gt_px = gt_obj.get('pixel_coords', [])
        if len(gt_px) < 2:
            continue
        # GT only has center pixel, no bbox — match by nearest center
        gt_cx, gt_cy = gt_px[0], gt_px[1]

        best_dist = float('inf')
        best_di = -1
        for di, det_obj in enumerate(det_objects):
            if di in used_det:
                continue
            bbox = det_obj.get('bbox', [])
            if len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = ((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_di = di

        # Accept match if center distance is reasonable (< 50px)
        if best_di >= 0 and best_dist < 50:
            pairs.append((gi, best_di))
            used_det.add(best_di)

    return pairs


def eval_relations(gt_scene: dict, det_scene: dict, pairs: List[Tuple[int, int]]) -> Dict:

    counts = {r: {'correct': 0, 'total': 0} for r in RELATION_TYPES}

    gt_objects = gt_scene.get('objects', [])
    det_objects = det_scene.get('objects', [])

    gt_idx_to_pair = {gi: di for gi, di in pairs}
    det_idx_to_pair = {di: gi for gi, di in pairs}

    for gi, di in pairs:
        gt_obj = gt_objects[gi]
        gt_relations = gt_obj.get('relations', {})

        for rel in RELATION_TYPES:
            gt_related_indices = [
                r['idx'] for r in gt_relations.get(rel, [])
                if r['idx'] < len(gt_objects)
            ]
            if not gt_related_indices:
                continue

            # For each GT relation, check if detected scene has the same relation
            det_obj = det_objects[di]
            det_relations = det_obj.get('relations', {})
            det_related_det_indices = [
                r['idx'] for r in det_relations.get(rel, [])
                if r['idx'] < len(det_objects)
            ]

            # Map detected indices back to GT indices
            det_related_gt_indices = [
                det_idx_to_pair[idx]
                for idx in det_related_det_indices
                if idx in det_idx_to_pair
            ]

            # Check overlap
            gt_set = set(gt_related_indices)
            det_set = set(det_related_gt_indices)

            counts[rel]['total'] += len(gt_set)
            counts[rel]['correct'] += len(gt_set & det_set)

    return counts


def run_perception_eval(jsonl_path: str, images_dir: str, limit: int = None) -> dict:
    from scene_graph.builder import SceneGraphBuilder
    builder = SceneGraphBuilder()

    samples = []
    with open(jsonl_path, encoding='utf-8') as f:
        seen_images = set()
        for line in f:
            s = json.loads(line)
            img_id = s['image_id']
            if img_id not in seen_images:
                seen_images.add(img_id)
                samples.append(s)
            if limit and len(samples) >= limit:
                break

    print(f'Evaluating perception on {len(samples)} images...')

    detection_stats = {'gt_total': 0, 'det_found': 0}
    count_by_scene_size = defaultdict(lambda: {'gt': 0, 'found': 0, 'scenes': 0})
    relation_stats = {r: {'correct': 0, 'total': 0} for r in RELATION_TYPES}

    for i, sample in enumerate(samples):
        img_path = Path(images_dir) / (sample['image_id'] + '.png')
        if not img_path.exists():
            # Try image_path from JSONL
            img_path = Path(sample['image_path'])
        if not img_path.exists():
            continue

        gt_scene = sample['scene_graph']
        gt_objects = gt_scene.get('objects', [])
        n_gt = len(gt_objects)

        try:
            det_scene = builder.build(str(img_path))
        except Exception as e:
            print(f'  Builder failed on {img_path.name}: {e}')
            continue

        det_objects = det_scene.get('objects', [])
        n_det = len(det_objects)

        # Detection recall
        detection_stats['gt_total'] += n_gt
        detection_stats['det_found'] += min(n_det, n_gt)  # can't find more than GT

        size_key = f'{n_gt} objects'
        count_by_scene_size[size_key]['gt'] += n_gt
        count_by_scene_size[size_key]['found'] += min(n_det, n_gt)
        count_by_scene_size[size_key]['scenes'] += 1

        # Relation accuracy (only if objects matched)
        pairs = match_objects(gt_objects, det_objects)
        if pairs and 'relations' in gt_objects[0]:
            rel_counts = eval_relations(gt_scene, det_scene, pairs)
            for rel in RELATION_TYPES:
                relation_stats[rel]['correct'] += rel_counts[rel]['correct']
                relation_stats[rel]['total'] += rel_counts[rel]['total']

        if (i + 1) % 25 == 0:
            recall = detection_stats['det_found'] / max(detection_stats['gt_total'], 1)
            print(f'  {i+1}/{len(samples)} — detection recall so far: {recall:.3f}')

    # Compile results
    det_recall = detection_stats['det_found'] / max(detection_stats['gt_total'], 1)
    rel_accuracy = {}
    for rel in RELATION_TYPES:
        t = relation_stats[rel]['total']
        c = relation_stats[rel]['correct']
        rel_accuracy[rel] = c / t if t > 0 else None

    results = {
        'n_images': len(samples),
        'detection_recall': det_recall,
        'detection_stats': detection_stats,
        'relation_accuracy': rel_accuracy,
        'relation_stats': relation_stats,
        'count_by_scene_size': dict(count_by_scene_size),
    }

    print(f'\n=== Perception Eval ===')
    print(f'Detection recall: {det_recall:.3f}')
    print(f'Relation accuracy:')
    for rel, acc in rel_accuracy.items():
        if acc is not None:
            print(f'  {rel:8s}: {acc:.3f}')
        else:
            print(f'  {rel:8s}: N/A (no GT relations)')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--images', default='datasets/clevr/images/val')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--output', default='results/perception_eval.json')
    args = parser.parse_args()

    results = run_perception_eval(args.jsonl, args.images, args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
