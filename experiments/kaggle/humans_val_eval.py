"""Evaluate the RL checkpoint with its original reward interpreter."""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import json, torch, subprocess
from pathlib import Path


MODEL_PATH = '/kaggle/input/datasets/leairt/t5-nl2dsl-rl-ckpt'
HUMANS_VAL = '/kaggle/input/datasets/leairt/clevr-humans-val/CLEVR-Humans-val.json'
VAL_SCENES = '/kaggle/input/datasets/leairt/clevr-val-scenes/CLEVR_val_scenes.json'

class SceneGraph:
    def __init__(self, d):
        self.objects = d.get('objects', [])
        self.rels    = d.get('relationships', {})
    def get_attribute(self, i, a): return self.objects[i][a]
    def get_related(self, i, rel):
        r = self.rels.get(rel, [])
        return r[i] if i < len(r) else []
    def all_objects(self): return list(range(len(self.objects)))

def _best(sg, o):
    return max(o, key=lambda i: (lambda b: (b[2]-b[0])*(b[3]-b[1]) if len(b)==4 else 0)(sg.objects[i].get('bbox',[0,0,0,0])))

def _unique(sg, o):
    if len(o) == 1: return o
    if len(o) == 0:
        a = sg.all_objects()
        if not a: raise ValueError('empty scene')
        return [_best(sg, a)]
    return [_best(sg, o)]

def _qattr(sg, o, attr):
    if not o: raise ValueError('empty')
    return sg.get_attribute(_best(sg,o) if len(o)>1 else o[0], attr)

def _same(sg, o, attr):
    if not o: raise ValueError('empty')
    ref = _best(sg,o) if len(o)>1 else o[0]
    val = sg.get_attribute(ref, attr)
    return [i for i in sg.all_objects() if i != ref and sg.get_attribute(i, attr) == val]

OPS = {
    'scene':          lambda sg,*a:   sg.all_objects(),
    'filter_color':   lambda sg,o,c:  [i for i in o if sg.get_attribute(i,'color')==c],
    'filter_shape':   lambda sg,o,s:  [i for i in o if sg.get_attribute(i,'shape')==s],
    'filter_size':    lambda sg,o,s:  [i for i in o if sg.get_attribute(i,'size')==s],
    'filter_material':lambda sg,o,m:  [i for i in o if sg.get_attribute(i,'material')==m],
    'relate':         lambda sg,o,r:  list({j for i in o for j in sg.get_related(i,r)}),
    'intersect':      lambda sg,a,b:  list(set(a)&set(b)),
    'union':          lambda sg,a,b:  list(set(a)|set(b)),
    'unique':         _unique,
    'count':          lambda sg,o:    len(o),
    'exist':          lambda sg,o:    len(o)>0,
    'query_color':    lambda sg,o:    _qattr(sg,o,'color'),
    'query_shape':    lambda sg,o:    _qattr(sg,o,'shape'),
    'query_size':     lambda sg,o:    _qattr(sg,o,'size'),
    'query_material': lambda sg,o:    _qattr(sg,o,'material'),
    'equal_color':    lambda sg,a,b:  a==b,
    'equal_shape':    lambda sg,a,b:  a==b,
    'equal_size':     lambda sg,a,b:  a==b,
    'equal_material': lambda sg,a,b:  a==b,
    'equal_integer':  lambda sg,a,b:  a==b,
    'less_than':      lambda sg,a,b:  a<b,
    'greater_than':   lambda sg,a,b:  a>b,
    'same_color':     lambda sg,o:    _same(sg,o,'color'),
    'same_shape':     lambda sg,o:    _same(sg,o,'shape'),
    'same_size':      lambda sg,o:    _same(sg,o,'size'),
    'same_material':  lambda sg,o:    _same(sg,o,'material'),
}

def execute(prog_str, sg):
    steps = [s.strip() for s in prog_str.strip().split('|')]
    if not steps or steps == ['']: return None, False, False
    result = None
    for step in steps:
        parts = step.split()
        if not parts or parts[0] not in OPS: return None, False, False
        args = ([sg] if result is None else [sg, result]) + parts[1:]
        try:    result = OPS[parts[0]](*args)
        except: return None, True, True
    return result, True, False

def norm(x):
    if x is None:          return 'none'
    if isinstance(x, bool): return 'yes' if x else 'no'
    return str(x).lower().strip()

def reward(prog, sg, gt):
    result, ok, err = execute(prog, sg)
    if not ok and not err: return 0.0
    if err:                return -0.1
    return 1.0 if norm(result) == norm(gt) else 0.2

print('Ucitavam...')
with open(VAL_SCENES) as f:
    scenes = {s['image_index']: s for s in json.load(f)['scenes']}
with open(HUMANS_VAL) as f:
    questions = json.load(f)['questions']
print(f'  {len(scenes)} scena, {len(questions)} pitanja')

device    = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f'Model ucitan sa {MODEL_PATH}')

total = correct = exec_err = parse_err = 0
by_type = {}

for i, q in enumerate(questions):
    if i % 200 == 0: print(f'  {i}/{len(questions)}...')
    sg  = SceneGraph(scenes.get(q['image_index'], {'objects': [], 'relationships': {}}))
    enc = tokenizer(q['question'], return_tensors='pt', max_length=64, truncation=True).to(device)
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=96, num_beams=4)
    prog = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    r    = reward(prog, sg, q['answer'])

    qt = q.get('question_type', 'other')
    if qt not in by_type: by_type[qt] = [0, 0]
    by_type[qt][1] += 1

    total += 1
    if r == 1.0:    correct  += 1; by_type[qt][0] += 1
    elif r == -0.1: exec_err += 1
    elif r == 0.0:  parse_err+= 1

print(f'\n{"="*50}')
print(f'CLEVR-Humans VAL — {total} pitanja')
print(f'{"="*50}')
print(f'Tacnost:      {correct}/{total} = {correct/total*100:.1f}%')
print(f'Exec errors:  {exec_err}/{total} = {exec_err/total*100:.1f}%')
print(f'Parse errors: {parse_err}/{total} = {parse_err/total*100:.1f}%')
print(f'\nPo tipu:')
for t, (c, n) in sorted(by_type.items()):
    print(f'  {t:<15} {c}/{n} = {c/n*100:.1f}%')
