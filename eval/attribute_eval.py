"""Evaluate CLIP attributes on crops around annotated CLEVR object centers."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLEVR_JSONL = 'datasets/converted/clevr/clevr_val.jsonl'

CROP_SIZE = {"large": 90, "small": 60}
DEFAULT_CROP = 75


def load_samples(jsonl_path, n):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if len(samples) >= n:
                break
            d = json.loads(line)
            if d.get('scene_graph', {}).get('objects'):
                samples.append(d)
    return samples


def bbox_from_pixel_coord(px, py, size_label):
    half = CROP_SIZE.get(size_label, DEFAULT_CROP) // 2
    return (px - half, py - half, px + half, py + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200, help='Broj pitanja (slika)')
    args = parser.parse_args()

    from perception.attribute_extractor import AttributeExtractor
    print('Ucitavam CLIP model...')
    extractor = AttributeExtractor()

    samples = load_samples(CLEVR_JSONL, args.n)
    print(f'Ucitano {len(samples)} pitanja')

    correct = defaultdict(int)
    total   = defaultdict(int)
    seen_images = set()
    obj_count = 0

    for sample in samples:
        img_path = sample['image_path']
        if img_path in seen_images:
            continue
        seen_images.add(img_path)

        if not Path(img_path).exists():
            continue

        image = Image.open(img_path).convert('RGB')
        objects = sample['scene_graph']['objects']

        for obj in objects:
            px, py = int(obj['pixel_coords'][0]), int(obj['pixel_coords'][1])
            size_label = obj.get('size', 'large')
            bbox = bbox_from_pixel_coord(px, py, size_label)

            try:
                attrs = extractor.extract_attributes(image, bbox=bbox,
                                                     attributes=['color', 'shape', 'material'])
            except Exception:
                continue

            for attr in ('color', 'shape', 'material'):
                gt_val  = obj.get(attr, '')
                pred_val = attrs.get(attr, {}).get('value', '')
                if gt_val:
                    total[attr] += 1
                    if pred_val == gt_val:
                        correct[attr] += 1

            obj_count += 1

    print(f'\nEvaluirano {obj_count} objekata sa {len(seen_images)} slika:\n')
    for attr in ('color', 'shape', 'material'):
        if total[attr]:
            acc = correct[attr] / total[attr]
            print(f'  {attr:<10} {correct[attr]:>4}/{total[attr]:<4}  = {acc:.1%}')


if __name__ == '__main__':
    main()
