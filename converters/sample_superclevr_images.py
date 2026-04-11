import json
import random
import shutil
from pathlib import Path

SEED = 42
N = 1000
JSONL_PATH = Path('datasets/converted/superclevr/superclevr_val.jsonl')
IMAGES_DIR = Path('datasets/super_clevr/images')
OUTPUT_DIR = Path('datasets/eval_sample_superclevr')


def main():
    # Sample images directly from the images folder — no need to read full JSONL first
    all_image_ids = [p.stem for p in IMAGES_DIR.glob('*.png')]
    print(f'Total images found: {len(all_image_ids)}')

    random.seed(SEED)
    selected_ids = set(random.sample(all_image_ids, min(N, len(all_image_ids))))
    print(f'Selected {len(selected_ids)} images (seed={SEED})')

    print('Filtering JSONL...')
    selected_samples = []
    with open(JSONL_PATH, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            if s['image_id'] in selected_ids:
                selected_samples.append(s)
            if len(selected_samples) > 0 and len(selected_samples) % 1000 == 0:
                print(f'  {len(selected_samples)} questions so far...')

    print(f'Questions for selected images: {len(selected_samples)}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_images = OUTPUT_DIR / 'images'
    out_images.mkdir(exist_ok=True)

    missing = []
    copied = 0
    for iid in selected_ids:
        src = IMAGES_DIR / f'{iid}.png'
        dst = out_images / f'{iid}.png'
        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(iid)

    jsonl_out = OUTPUT_DIR / 'superclevr_eval.jsonl'
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for s in selected_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    ids_file = OUTPUT_DIR / 'image_ids.txt'
    with open(ids_file, 'w') as f:
        f.write('\n'.join(sorted(selected_ids)))

    print(f'\nResult:')
    print(f'  Images copied:  {copied}/{len(selected_ids)}')
    if missing:
        print(f'  Missing:        {len(missing)} images')
    print(f'  Questions:      {len(selected_samples)}')
    print(f'\nFolder ready for upload: {OUTPUT_DIR.absolute()}')


if __name__ == '__main__':
    main()
