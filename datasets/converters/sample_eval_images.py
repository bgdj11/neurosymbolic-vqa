import json
import random
import shutil
from pathlib import Path

SEED = 42
N = 1000
JSONL_PATH = Path('datasets/converted/clevr/clevr_val.jsonl')
IMAGES_DIR = Path('datasets/clevr/images/val')
OUTPUT_DIR = Path('datasets/eval_sample')


def main():
    print('Ucitavam JSONL...')
    all_samples = []
    image_ids_ordered = []
    seen = set()

    with open(JSONL_PATH, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            all_samples.append(s)
            iid = s['image_id']
            if iid not in seen:
                seen.add(iid)
                image_ids_ordered.append(iid)

    print(f'Ukupno pitanja: {len(all_samples)}')
    print(f'Ukupno jedinstvenih slika: {len(image_ids_ordered)}')

    # Nasumican odabir slika
    random.seed(SEED)
    selected_ids = set(random.sample(image_ids_ordered, min(N, len(image_ids_ordered))))
    print(f'Izabrano {len(selected_ids)} slika (seed={SEED})')

    # Filtriraj pitanja za izabrane slike
    selected_samples = [s for s in all_samples if s['image_id'] in selected_ids]
    print(f'Pitanja za izabrane slike: {len(selected_samples)}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_images = OUTPUT_DIR / 'images'
    out_images.mkdir(exist_ok=True)

    # Kopiraj slike
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

    # Sacuvaj filtrirani JSONL
    jsonl_out = OUTPUT_DIR / 'clevr_val_eval.jsonl'
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for s in selected_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # Sacuvaj listu image IDs
    ids_file = OUTPUT_DIR / 'image_ids.txt'
    with open(ids_file, 'w') as f:
        f.write('\n'.join(sorted(selected_ids)))

    print(f'\nRezultat:')
    print(f'  Slike kopirane:   {copied}/{len(selected_ids)}')
    if missing:
        print(f'  Nedostaje:        {len(missing)} slika')
    print(f'  Pitanja u JSONL:  {len(selected_samples)}')
    print(f'\nFolder za upload na Drive:')
    print(f'  {(OUTPUT_DIR / "images").absolute()}')
    print(f'  {jsonl_out.absolute()}')
    print(f'\nimage_ids.txt commituj u repo:')
    print(f'  {ids_file.absolute()}')


if __name__ == '__main__':
    main()
