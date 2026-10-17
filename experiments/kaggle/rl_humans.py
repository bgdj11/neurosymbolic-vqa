"""REINFORCE experiment with the original sequential reward interpreter."""

# %%
import subprocess
print('OK')


# %%
from pathlib import Path

def find_input(filename):
    """Nadji fajl u /kaggle/input/ bez obzira na naziv dataseta."""
    for p in Path('/kaggle/input').rglob(filename):
        return str(p)
    return None

MODEL_IN     = find_input('config.json')
for p in Path('/kaggle/input').rglob('config.json'):
    if (p.parent / 'tokenizer_config.json').exists():
        MODEL_IN = str(p.parent)
        break

HUMANS_TRAIN = find_input('CLEVR-Humans-train.json')
VAL_SCENES   = find_input('CLEVR_val_scenes.json')
MODEL_OUT    = '/kaggle/working/t5-nl2dsl-rl'
CKPT_DIR     = '/kaggle/working/rl_checkpoints'

print(f'Model:        {MODEL_IN}')
print(f'Humans train: {HUMANS_TRAIN}')
print(f'Val scenes:   {VAL_SCENES}')

assert MODEL_IN,     'GRESKA: t5 model nije nadjen u /kaggle/input/'
assert HUMANS_TRAIN, 'GRESKA: CLEVR-Humans-train.json nije nadjen'


# %%

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
    """Vraca (result, executed_ok, exec_error)."""
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
    if x is None:         return 'none'
    if isinstance(x,bool):return 'yes' if x else 'no'
    return str(x).lower().strip()

def reward(prog, sg, gt):
    """
    +1.0  tacan odgovor
    +0.2  izvrsivo ali pogresno  (validan DSL = korisno)
    -0.1  execution error        (kazna za invalidan program)
     0.0  parse error
    """
    result, ok, err = execute(prog, sg)
    if not ok and not err: return 0.0
    if err:                return -0.1
    return 1.0 if norm(result) == norm(gt) else 0.2

print('DSL engine OK')


# %%
import json, random

TRAIN_SCENES = find_input('CLEVR_train_scenes.json')
assert TRAIN_SCENES, 'GRESKA: CLEVR_train_scenes.json nije nadjen (potreban za Humans train split)'

print('Ucitavam train_scenes (moze potrajati ~1min)...')
with open(TRAIN_SCENES) as f:
    scenes_by_idx = {s['image_index']: s for s in json.load(f)['scenes']}
print(f'  {len(scenes_by_idx)} train scena')

print('Ucitavam CLEVR-Humans train...')
with open(HUMANS_TRAIN) as f:
    raw = json.load(f)['questions']

train_rows, skipped = [], 0
for q in raw:
    idx = q['image_index']
    if idx not in scenes_by_idx: skipped += 1; continue
    train_rows.append({'question': q['question'], 'answer': q['answer'], 'scene': scenes_by_idx[idx]})

print(f'  {len(train_rows)} pitanja ucitano, {skipped} preskoceno')
random.seed(42)
random.shuffle(train_rows)

eval_rows  = train_rows[-500:]
train_rows = train_rows[:-500]
print(f'  Train: {len(train_rows)}, Eval: {len(eval_rows)}')


# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

print('\nUcitavam model za coverage analizu...')
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(MODEL_IN)
tmp_model = T5ForConditionalGeneration.from_pretrained(MODEL_IN).to(device)
tmp_model.eval()

N_COV = 300
covered = correct = execable = 0
for row in train_rows[:N_COV]:
    sg  = SceneGraph(row['scene'])
    enc = tokenizer(row['question'], return_tensors='pt', max_length=64, truncation=True).to(device)
    with torch.no_grad():
        seqs = tmp_model.generate(**enc, max_new_tokens=96, num_beams=5, num_return_sequences=5, early_stopping=True)
    best_r = max(reward(tokenizer.decode(s, skip_special_tokens=True).strip(), sg, row['answer']) for s in seqs)
    if best_r == 1.0:  covered += 1
    if best_r >= 0.2:  execable += 1

print(f'\n=== Coverage (n={N_COV}) ===')
print(f'Tacan odgovor (bar 1 beam):  {covered}/{N_COV} = {covered/N_COV*100:.1f}%')
print(f'Izvrsiv (bar 1 beam):        {execable}/{N_COV} = {execable/N_COV*100:.1f}%')
print(f'Procena na {len(train_rows)} pitanja:  ~{int(covered/N_COV*len(train_rows))} DSL-pokrivenih')

del tmp_model; torch.cuda.empty_cache()


# %%
import torch.nn.functional as F
from torch.optim import AdamW
import time, zipfile

N_SAMPLES  = 4
BATCH_Q    = 4
LR         = 8e-6
GRAD_CLIP  = 1.0
KL_BETA    = 0.05
EVAL_EVERY = 400
CKPT_EVERY = 1000
MAX_HOURS  = 8.5
GEN_TOKENS = 96
TEMPERATURE= 1.1

def save_zip_checkpoint(step, model, tokenizer):
    """Sacuvaj model i odmah zipuj — fajl je downloadable iz Output taba."""
    ckpt = f'{CKPT_DIR}/step{step:05d}'
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    zip_path = f'{CKPT_DIR}/step{step:05d}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in Path(ckpt).rglob('*'):
            if f.is_file():
                zf.write(f, arcname=f.name)
    print(f'[Checkpoint] {zip_path} ({Path(zip_path).stat().st_size/1e6:.0f}MB)')

print('Ucitavam modele...')
model     = T5ForConditionalGeneration.from_pretrained(MODEL_IN).to(device)
ref_model = T5ForConditionalGeneration.from_pretrained(MODEL_IN).to(device)
ref_model.eval()
for p in ref_model.parameters(): p.requires_grad_(False)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

print('Cuva step0 (verifikacija)...')
save_zip_checkpoint(0, model, tokenizer)
print('OK — zip sacuvan, nastavljam trening')

def run_eval(rows, n=200):
    model.eval()
    sample = random.sample(rows, min(n, len(rows)))
    correct = execable = 0
    for row in sample:
        sg  = SceneGraph(row['scene'])
        enc = tokenizer(row['question'], return_tensors='pt', max_length=64, truncation=True).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=GEN_TOKENS, num_beams=4)
        r = reward(tokenizer.decode(ids[0], skip_special_tokens=True).strip(), sg, row['answer'])
        if r == 1.0: correct += 1
        if r >= 0.2: execable += 1
    model.train()
    return correct/len(sample)*100, execable/len(sample)*100

print('\n' + '='*60)
print(f'REINFORCE | LR={LR} | N_SAMPLES={N_SAMPLES} | KL_BETA={KL_BETA}')
print(f'Train: {len(train_rows)} pitanja | Max: {MAX_HOURS}h')
print('='*60)

model.train()
t0 = time.time()
step = epoch = data_idx = 0
reward_log, acc_log = [], []

while True:
    if (time.time()-t0)/3600 >= MAX_HOURS:
        print(f'\n[Stop] {(time.time()-t0)/3600:.1f}h')
        break

    if data_idx >= len(train_rows):
        epoch += 1; data_idx = 0; random.shuffle(train_rows)
        print(f'\n── Epoha {epoch} ──')

    batch    = train_rows[data_idx: data_idx + BATCH_Q]
    data_idx += BATCH_Q

    rl_terms  = []
    kl_terms  = []
    all_rewards = []

    for row in batch:
        sg  = SceneGraph(row['scene'])
        enc = tokenizer(row['question'], return_tensors='pt', max_length=64, truncation=True).to(device)
        inp, mask = enc['input_ids'], enc['attention_mask']

        rewards, log_probs = [], []

        for _ in range(N_SAMPLES):
            with torch.no_grad():
                ids = model.generate(inp, attention_mask=mask,
                                     max_new_tokens=GEN_TOKENS,
                                     do_sample=True, temperature=TEMPERATURE, top_p=0.95)
            prog = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            r    = reward(prog, sg, row['answer'])
            rewards.append(r)

            labels = ids[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            log_probs.append(-out.loss)

        baseline = sum(rewards) / len(rewards)
        all_rewards.extend(rewards)

        for r, lp in zip(rewards, log_probs):
            rl_terms.append(-(r - baseline) * lp)

        if KL_BETA > 0:
            dec0 = torch.zeros((1,1), dtype=torch.long, device=device)
            with torch.no_grad():
                ref_logits = ref_model(inp, attention_mask=mask, decoder_input_ids=dec0).logits
            cur_logits = model(inp, attention_mask=mask, decoder_input_ids=dec0).logits
            kl = F.kl_div(F.log_softmax(cur_logits,-1),
                          F.softmax(ref_logits.detach(),-1), reduction='batchmean')
            kl_terms.append(kl)

    n = len(batch) * N_SAMPLES
    loss = sum(rl_terms) / n
    if kl_terms:
        loss = loss + KL_BETA * sum(kl_terms) / len(kl_terms)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    step += 1
    reward_log.append(sum(all_rewards)/len(all_rewards))

    if step % 50 == 0:
        avg_r = sum(reward_log[-100:])/len(reward_log[-100:])
        print(f'Step {step:5d} | ep={epoch} | '
              f'reward={avg_r:.3f} | loss={loss.item():.4f} | '
              f'{(time.time()-t0)/3600:.1f}h/{MAX_HOURS}h')

    if step % EVAL_EVERY == 0:
        acc, ex = run_eval(eval_rows)
        acc_log.append((step, acc, ex))
        print(f'\n>>> EVAL step={step}: acc={acc:.1f}%  exec={ex:.1f}% <<<\n')

    if step % CKPT_EVERY == 0:
        save_zip_checkpoint(step, model, tokenizer)


# %%
model.save_pretrained(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
print(f'\nModel sacuvan: {MODEL_OUT}')

print('\n=== Eval istorija ===')
print(f'{"Step":>6}  {"Acc%":>6}  {"Exec%":>6}')
for s, a, e in acc_log:
    print(f'{s:6d}  {a:6.1f}  {e:6.1f}')

if reward_log:
    n_seg = 10
    seg = max(1, len(reward_log)//n_seg)
    print('\n=== Reward po segmentima ===')
    for i in range(n_seg):
        chunk = reward_log[i*seg:(i+1)*seg]
        if chunk: print(f'  {i*seg}-{(i+1)*seg}: {sum(chunk)/len(chunk):.3f}')

print('\nSkini t5-nl2dsl-rl/ i rl_checkpoints/ iz Output panela.')
