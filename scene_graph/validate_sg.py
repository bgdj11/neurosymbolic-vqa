"""Compare detected and annotated graphs using nearest-center matching."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

JSONL    = 'datasets/converted/clevr/clevr_val.jsonl'
N        = 50
MATCH_PX = 60


def load_unique_images(jsonl_path: str, n: int) -> list[dict]:
    seen, samples = set(), []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            img = s.get('image_path', '')
            if img not in seen and Path(img).exists():
                seen.add(img)
                samples.append(s)
            if len(samples) >= n:
                break
    return samples


def match_objects(gt_objs: list, det_objs: list) -> list[tuple]:
    """
    Vraca listu (gt_idx, det_idx) parova.
    Svaki GT objekat se spaja sa najblizim detektovanim koji jos nije uzet.
    """
    if not gt_objs or not det_objs:
        return []

    def px(obj):
        c = obj.get('pixel_coords', [0, 0])
        return np.array([c[0], c[1]], dtype=float)

    used_det = set()
    pairs = []
    for gi, go in enumerate(gt_objs):
        best_di, best_d = None, float('inf')
        for di, do in enumerate(det_objs):
            if di in used_det:
                continue
            d = np.linalg.norm(px(go) - px(do))
            if d < best_d:
                best_d, best_di = d, di
        if best_di is not None and best_d <= MATCH_PX:
            pairs.append((gi, best_di))
            used_det.add(best_di)
    return pairs


def relation_accuracy(gt_sg: dict, det_sg: dict, pairs: list) -> dict[str, tuple]:
    """
    Za svaki par matched objekata proveri da li je relacija sacuvana.
    Vraca {direction: (correct, total)}.
    """
    gt_rels  = gt_sg.get('relationships', {})
    det_rels = det_sg.get('relationships', {})
    dirs = ['left', 'right', 'front', 'behind']
    stats = {d: [0, 0] for d in dirs}

    gt_to_det = {gi: di for gi, di in pairs}
    det_to_gt = {di: gi for gi, di in pairs}

    for direction in dirs:
        gt_dir  = gt_rels.get(direction, [])
        det_dir = det_rels.get(direction, [])

        for gi, di in pairs:
            if gi >= len(gt_dir):
                continue
            gt_neighbors_gt_idx  = set(gt_dir[gi])
            gt_neighbors_det_idx = {gt_to_det[x] for x in gt_neighbors_gt_idx if x in gt_to_det}

            if di >= len(det_dir):
                stats[direction][1] += 1
                continue
            det_neighbors_det_idx = set(det_dir[di])

            stats[direction][1] += 1
            if gt_neighbors_det_idx == det_neighbors_det_idx:
                stats[direction][0] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',     type=int, default=N,    help='Broj slika')
    parser.add_argument('--jsonl', default=JSONL,          help='JSONL dataset')
    args = parser.parse_args()

    print(f'Ucitavam SceneGraphBuilder...')
    from scene_graph.builder import SceneGraphBuilder
    builder = SceneGraphBuilder()

    samples = load_unique_images(args.jsonl, args.n)
    print(f'Slika za evaluaciju: {len(samples)}\n')

    count_correct = count_total = 0
    attr_correct  = defaultdict(int)
    attr_total    = defaultdict(int)
    rel_correct   = defaultdict(int)
    rel_total     = defaultdict(int)
    matched_objs  = unmatched_gt = unmatched_det = 0
    build_errors  = 0

    for i, sample in enumerate(samples):
        img_path = sample['image_path']
        gt_sg    = sample['scene_graph']
        gt_objs  = gt_sg.get('objects', [])

        try:
            det_sg   = builder.build(img_path)
            det_objs = det_sg.get('objects', [])
        except Exception as e:
            print(f'  [{i+1}] GRESKA pri gradnji: {e}')
            build_errors += 1
            continue

        count_total += 1
        if len(gt_objs) == len(det_objs):
            count_correct += 1

        pairs = match_objects(gt_objs, det_objs)
        matched_objs  += len(pairs)
        unmatched_gt  += len(gt_objs)  - len(pairs)
        unmatched_det += len(det_objs) - len(pairs)

        for gi, di in pairs:
            go, do = gt_objs[gi], det_objs[di]
            for attr in ('color', 'shape', 'material', 'size'):
                gt_val  = go.get(attr, '')
                det_val = do.get(attr, '')
                if gt_val:
                    attr_total[attr]   += 1
                    if det_val == gt_val:
                        attr_correct[attr] += 1

        rel_stats = relation_accuracy(gt_sg, det_sg, pairs)
        for d, (c, t) in rel_stats.items():
            rel_correct[d] += c
            rel_total[d]   += t

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1:>3}/{len(samples)}] obj_count_acc={count_correct/max(count_total,1):.1%}  '
                  f'matched={matched_objs}  build_err={build_errors}')

    print(f'\n{"="*55}')
    print(f'SCENE GRAPH VALIDATION — {len(samples)} slika')
    print(f'{"="*55}')
    print(f'\n[DETEKCIJA OBJEKATA]')
    print(f'  Tacan broj objekata: {count_correct}/{count_total} = {count_correct/max(count_total,1):.1%}')
    print(f'  Matchovanih parova:  {matched_objs}')
    print(f'  GT bez para:         {unmatched_gt}  (propusteni objekti)')
    print(f'  Det bez para:        {unmatched_det} (lazni pozitivi)')
    print(f'  Greske pri gradnji:  {build_errors}')

    print(f'\n[ATRIBUTI] (na matchovanim objektima)')
    for attr in ('color', 'shape', 'material', 'size'):
        t = attr_total[attr]
        c = attr_correct[attr]
        bar = '█' * int(c/max(t,1)*20) + '░' * (20 - int(c/max(t,1)*20))
        print(f'  {attr:<10} [{bar}] {c}/{t} = {c/max(t,1):.1%}')

    print(f'\n[PROSTORNE RELACIJE] (na matchovanim parovima)')
    for d in ('left', 'right', 'front', 'behind'):
        t = rel_total[d]
        c = rel_correct[d]
        print(f'  {d:<8} {c}/{t} = {c/max(t,1):.1%}')

    total_attr_c = sum(attr_correct.values())
    total_attr_t = sum(attr_total.values())
    total_rel_c  = sum(rel_correct.values())
    total_rel_t  = sum(rel_total.values())
    print(f'\n[UKUPNO]')
    print(f'  Atributi:  {total_attr_c}/{total_attr_t} = {total_attr_c/max(total_attr_t,1):.1%}')
    print(f'  Relacije:  {total_rel_c}/{total_rel_t} = {total_rel_c/max(total_rel_t,1):.1%}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
