# ─ Cell 1
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, time
from pathlib import Path

DRIVE_BASE = '/content/drive/MyDrive'
REPO_DIR = '/content/neurosymbolic-vqa'
RESULTS_DIR = f'{DRIVE_BASE}/results'
SCENES_DIR = f'{DRIVE_BASE}/results/scene_graphs'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR,  exist_ok=True)


# ── Cell 2
if not os.path.exists(REPO_DIR):
    import zipfile
    repo_zip = f'{DRIVE_BASE}/neurosymbolic-vqa.zip'
    if not os.path.exists(repo_zip):
        raise FileNotFoundError(f'Upload neurosymbolic-vqa.zip to Drive root first.')
    with zipfile.ZipFile(repo_zip, 'r') as z:
        z.extractall('/content')
    print('Repo extracted.')

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print('Repo ready.')


# ── Cell 3
import subprocess
subprocess.run(['pip', 'install', '-q',
    'transformers', 'torch', 'torchvision',
    'ultralytics',
    'git+https://github.com/CASIA-IVA-Lab/FastSAM.git',
    'timm', 'open-clip-torch', 'tqdm',
], check=False)
print('Dependencies installed.')


# ── Cell 4
MODEL_DST = f'{REPO_DIR}/models/t5-nl2dsl-final'
JSONL_DST = f'{REPO_DIR}/datasets/eval_sample/clevr_val_eval.jsonl'
IMAGES_DST = f'{REPO_DIR}/datasets/eval_sample/images'

assert os.path.exists(MODEL_DST),  f'Missing: {MODEL_DST}'
assert os.path.exists(JSONL_DST),  f'Missing: {JSONL_DST}'
assert os.path.exists(IMAGES_DST), f'Missing: {IMAGES_DST}'
n_images = len(list(Path(IMAGES_DST).glob('*.png')))
print(f'Model OK | JSONL OK | {n_images} images OK')


# ── Cell 5
DATASET = 'clevr'
LIMIT = None
CHECKPOINT_FILE = f'{RESULTS_DIR}/{DATASET}_detected_checkpoint.jsonl'

samples = []
with open(JSONL_DST, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if LIMIT and i >= LIMIT:
            break
        samples.append(json.loads(line))

done = {}
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            done[r['question_id']] = r
    print(f'Checkpoint: {len(done)} questions already done.')

todo = [s for s in samples if s['question_id'] not in done]
print(f'Remaining: {len(todo)} questions, '
      f'{len(set(s["image_id"] for s in todo))} unique images.')


# ── Cell 6
from scene_graph.builder import SceneGraphBuilder
from tqdm.auto import tqdm

builder = SceneGraphBuilder()

sg_cache = {}
for sg_file in Path(SCENES_DIR).glob(f'{DATASET}_*.json'):
    img_id = sg_file.stem[len(DATASET) + 1:]
    with open(sg_file) as fp:
        sg_cache[img_id] = json.load(fp)
print(f'Pre-loaded {len(sg_cache)} scene graphs from Drive.')

todo_images = {s['image_id']: s['image_path'] for s in todo
               if s['image_id'] not in sg_cache}

print(f'Building {len(todo_images)} new scene graphs...')
for img_id, img_path in tqdm(todo_images.items(), desc='Perception'):
    local_path = f'{IMAGES_DST}/{Path(img_path).name}'
    if not os.path.exists(local_path):
        sg_cache[img_id] = None
        continue
    try:
        sg = builder.build(local_path)
        sg_cache[img_id] = sg
        # Save to Drive — small JSON, reusable for thesis viz later
        with open(f'{SCENES_DIR}/{DATASET}_{img_id}.json', 'w') as fp:
            json.dump(sg, fp)
    except Exception as e:
        print(f'  {img_id}: {e}')
        sg_cache[img_id] = None

ok = sum(1 for v in sg_cache.values() if v is not None)
print(f'Scene graphs ready: {ok}/{len(sg_cache)}')


# ── Cell 7
from nl2dsl.infer import NL2DSLModel
from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph

BATCH_SIZE = 32

print('Loading model...')
model = NL2DSLModel(MODEL_DST)

ckpt_file = open(CHECKPOINT_FILE, 'a', encoding='utf-8')
results_new = []
t0 = time.time()

for start in tqdm(range(0, len(todo), BATCH_SIZE), desc='NL2DSL+Exec'):
    batch = todo[start:start + BATCH_SIZE]
    programs = model.predict_batch([s['question'] for s in batch], batch_size=BATCH_SIZE)

    for sample, program in zip(batch, programs):
        img_id = sample['image_id']
        det_sg = sg_cache.get(img_id)

        if program is None:
            r = dict(question_id=sample['question_id'], question=sample['question'],
                     expected_answer=sample['answer'], predicted_answer=None,
                     correct=False, track='detected', program_status='failed',
                     execution_error=None, image_id=img_id)
        elif det_sg is None:
            r = dict(question_id=sample['question_id'], question=sample['question'],
                     expected_answer=sample['answer'], predicted_answer=None,
                     correct=False, track='detected', program_status='valid',
                     execution_error='image_not_found', image_id=img_id)
        else:
            try:
                answer = str(Executor(SceneGraph(det_sg)).execute(program)).lower()
                r = dict(question_id=sample['question_id'], question=sample['question'],
                         expected_answer=sample['answer'], predicted_answer=answer,
                         correct=answer == str(sample['answer']).lower(),
                         track='detected', program_status='valid',
                         execution_error=None, image_id=img_id)
            except ExecutionError as e:
                r = dict(question_id=sample['question_id'], question=sample['question'],
                         expected_answer=sample['answer'], predicted_answer=None,
                         correct=False, track='detected', program_status='valid',
                         execution_error=str(e), image_id=img_id)

        results_new.append(r)
        ckpt_file.write(json.dumps(r) + '\n')

    ckpt_file.flush()
    done_n = len(done) + len(results_new)
    total_n = len(done) + len(todo)
    correct = sum(1 for r in results_new if r['correct'])
    print(f'  {done_n}/{total_n} | acc={correct/max(len(results_new),1):.3f} | {time.time()-t0:.0f}s')

ckpt_file.close()

# ── Cell 8
all_results = list(done.values()) + results_new
total   = len(all_results)
correct = sum(1 for r in all_results if r['correct'])
valid   = sum(1 for r in all_results if r['program_status'] == 'valid')
exec_err = sum(1 for r in all_results
               if r.get('execution_error') not in (None, 'image_not_found'))

cats = {}
for r in all_results:
    c = r.get('category', 'other')
    cats.setdefault(c, []).append(r['correct'])

summary = {
    'track':    'detected',
    'dataset':  DATASET,
    'total':    total,
    'correct':  correct,
    'accuracy': round(correct / total, 4) if total else 0,
    'accuracy_by_category': {c: round(sum(v)/len(v), 4) for c, v in cats.items()},
    'valid_program_rate':   round(valid / total, 4) if total else 0,
    'execution_error_rate': round(exec_err / total, 4) if total else 0,
}

import pprint
pprint.pprint(summary)

with open(f'{RESULTS_DIR}/{DATASET}_detected_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

with open(f'{RESULTS_DIR}/{DATASET}_detected_per_question.jsonl', 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

print('Saved to Drive.')

