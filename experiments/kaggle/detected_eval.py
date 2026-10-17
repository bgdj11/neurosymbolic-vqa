"""Evaluate detected scene graphs in the cloud runtime."""

# %%
import subprocess




print('Paketi instalirani.')


# %%
import sys, os, json, time, shutil
from pathlib import Path

BASE      = '/kaggle/input/datasets/bogdandjukic/neurosymbolic-vqa-eval-test'
CODE_DIR  = f'{BASE}/neurosymbolic-vqa'
JSONL     = f'{BASE}/clevr_val_eval.jsonl'
IMAGES    = f'{BASE}/images/images'
MODEL_DIR = '/kaggle/input/datasets/bogdandjukic/t5-nl2dsl-final/t5-nl2dsl-final'
CKPT      = '/kaggle/working/clevr_detected_checkpoint.jsonl'

WORK_CODE = '/kaggle/working/code'
shutil.copytree(CODE_DIR, WORK_CODE, dirs_exist_ok=True)
sys.path.insert(0, WORK_CODE)
os.chdir(WORK_CODE)

print('Slike:',  len(list(Path(IMAGES).glob('*.png'))))
print('JSONL:',  Path(JSONL).exists())
print('Model:',  Path(MODEL_DIR).exists())


# %%
if Path(CKPT).exists():
    raise FileExistsError(f'Output already exists: {CKPT}. Choose a new CKPT for a fresh run.')

samples = []
with open(JSONL, encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Ucitano {len(samples)} pitanja.')

from scene_graph.builder import SceneGraphBuilder
from tqdm.auto import tqdm

builder  = SceneGraphBuilder()
sg_cache = {}
t0       = time.time()

for img_path in tqdm(set(s['image_path'] for s in samples), desc='Percepcija'):
    local = f'{IMAGES}/{Path(img_path).name}'
    try:
        sg_cache[img_path] = builder.build(local) if Path(local).exists() else None
    except Exception as e:
        print(f'  {Path(img_path).name}: {e}')
        sg_cache[img_path] = None

ok = sum(1 for v in sg_cache.values() if v is not None)
print(f'Scene grafovi: {ok}/{len(sg_cache)} ({time.time()-t0:.0f}s)')

from nl2dsl.infer import NL2DSLModel
from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph
from eval.metrics import categorize_question

BATCH = 32
print('Ucitavam T5 model...')
model      = NL2DSLModel(MODEL_DIR)
ckpt_f     = open(CKPT, 'a', encoding='utf-8')
results_new = []
t0          = time.time()

for start in tqdm(range(0, len(samples), BATCH), desc='NL2DSL+Exec'):
    batch    = samples[start:start + BATCH]
    programs = model.predict_batch([s['question'] for s in batch], batch_size=BATCH)

    for sample, program in zip(batch, programs):
        det_sg = sg_cache.get(sample['image_path'])

        if program is None:
            r = dict(question_id=sample['question_id'], question=sample['question'],
                     expected_answer=sample['answer'], predicted_answer=None,
                     correct=False, track='detected', program_status='failed', execution_error=None)
        elif det_sg is None:
            r = dict(question_id=sample['question_id'], question=sample['question'],
                     expected_answer=sample['answer'], predicted_answer=None,
                     correct=False, track='detected', program_status='valid', execution_error='image_not_found')
        else:
            try:
                ans = str(Executor(SceneGraph(det_sg)).execute(program)).lower()
                r = dict(question_id=sample['question_id'], question=sample['question'],
                         expected_answer=sample['answer'], predicted_answer=ans,
                         correct=ans == str(sample['answer']).lower(),
                         track='detected', program_status='valid', execution_error=None)
            except ExecutionError as e:
                r = dict(question_id=sample['question_id'], question=sample['question'],
                         expected_answer=sample['answer'], predicted_answer=None,
                         correct=False, track='detected', program_status='valid', execution_error=str(e))

        r['category'] = categorize_question(sample['question'])
        results_new.append(r)
        ckpt_f.write(json.dumps(r) + '\n')

    ckpt_f.flush()
    acc = sum(r['correct'] for r in results_new) / len(results_new)
    print(f'  {len(results_new)}/{len(samples)} | acc={acc:.3f} | {time.time()-t0:.0f}s')

ckpt_f.close()


# %%
all_results = results_new
total    = len(all_results)
correct  = sum(r['correct'] for r in all_results)
valid    = sum(r['program_status'] == 'valid' for r in all_results)
exec_err = sum(1 for r in all_results if r.get('execution_error') not in (None, 'image_not_found'))

cats = {}
for r in all_results:
    c = r.get('category', 'other')
    cats.setdefault(c, []).append(r['correct'])

summary = {
    'track':    'detected',
    'dataset':  'clevr',
    'total':    total,
    'correct':  correct,
    'accuracy': round(correct / total, 4) if total else 0,
    'accuracy_by_category': {c: round(sum(v)/len(v), 4) for c, v in cats.items()},
    'valid_program_rate':   round(valid / total, 4) if total else 0,
    'execution_error_rate': round(exec_err / total, 4) if total else 0,
}

import pprint
pprint.pprint(summary)

with open('/kaggle/working/clevr_detected_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Sacuvano u Output panelu.')
