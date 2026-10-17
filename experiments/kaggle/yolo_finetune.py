"""Fine-tune YOLOv8m using approximate CLEVR bounding-box labels."""

# %%
import subprocess
print('OK')


# %%
import json, os, shutil
from pathlib import Path
import yaml

CLEVR_DIR = '/kaggle/input/datasets/timoboz/clevr-dataset/CLEVR_v1.0'

TRAIN_IMGS   = f'{CLEVR_DIR}/images/train'
VAL_IMGS     = f'{CLEVR_DIR}/images/val'
TRAIN_SCENES = f'{CLEVR_DIR}/scenes/CLEVR_train_scenes.json'
VAL_SCENES   = f'{CLEVR_DIR}/scenes/CLEVR_val_scenes.json'

OUT  = '/kaggle/working/clevr_yolo'
WOUT = '/kaggle/working'

IMG_W, IMG_H = 480, 320
BBOX_PX = {'small': 58, 'large': 92}

N_TRAIN = 35000
N_VAL   = 500

print('Train images:', Path(TRAIN_IMGS).exists())
print('Train scenes:', Path(TRAIN_SCENES).exists())
print('Val images:  ', Path(VAL_IMGS).exists())


# %%
def make_labels(scenes_json, img_dir, labels_dir, n_max=None):
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    with open(scenes_json) as f:
        scenes = json.load(f)['scenes']
    if n_max:
        scenes = scenes[:n_max]

    ok = 0
    for s in scenes:
        img_name = s['image_filename']
        if not (Path(img_dir) / img_name).exists():
            continue

        lines = []
        for obj in s['objects']:
            px = float(obj['pixel_coords'][0])
            py = float(obj['pixel_coords'][1])
            bpx = BBOX_PX.get(obj.get('size', 'small'), 65)

            cx = px / IMG_W
            cy = py / IMG_H
            w  = bpx / IMG_W
            h  = bpx / IMG_H

            cx = max(w / 2, min(1 - w / 2, cx))
            cy = max(h / 2, min(1 - h / 2, cy))
            lines.append(f'0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}')

        label_path = Path(labels_dir) / img_name.replace('.png', '.txt')
        label_path.write_text('\n'.join(lines))
        ok += 1

    print(f'  Labele: {ok}/{len(scenes)}')

print('Train labele...')
make_labels(TRAIN_SCENES, TRAIN_IMGS, f'{OUT}/labels/train', N_TRAIN)
print('Val labele...')
make_labels(VAL_SCENES,   VAL_IMGS,   f'{OUT}/labels/val',   N_VAL)

from tqdm.auto import tqdm

for split, src_dir in [('train', TRAIN_IMGS), ('val', VAL_IMGS)]:
    dst_dir = Path(f'{OUT}/images/{split}')
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = [f.stem for f in Path(f'{OUT}/labels/{split}').glob('*.txt')]
    print(f'Kopiranje {len(names)} {split} slika...')
    for name in tqdm(names):
        src = Path(src_dir) / f'{name}.png'
        dst = dst_dir / f'{name}.png'
        if not dst.exists() and src.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f'  {split}: {len(list(dst_dir.glob("*.png")))} slika gotovo')

cfg = {
    'path':  OUT,
    'train': 'images/train',
    'val':   'images/val',
    'nc':    1,
    'names': ['object'],
}
yaml_path = f'{OUT}/clevr.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(cfg, f)
print(f'Dataset YAML: {yaml_path}')


# %%
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data      = yaml_path,
    epochs    = 15,
    imgsz     = 480,
    batch     = 64,
    lr0       = 0.005,
    lrf       = 0.01,
    warmup_epochs = 2,
    patience    = 0,
    save_period = 3,
    val         = False,
    half        = True,
    device      = 0,
    project     = WOUT,
    name        = 'yolo_clevr',
    exist_ok    = True,
    plots       = False,
    verbose     = False,
)

best = f'{WOUT}/yolo_clevr/weights/best.pt'
shutil.copy(best, f'{WOUT}/yolov8m_clevr.pt')
print(f'\nModel sacuvan: {WOUT}/yolov8m_clevr.pt')
print('Skini iz Output panela.')
