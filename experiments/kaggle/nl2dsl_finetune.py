"""Fine-tune T5 on execution-filtered pseudo-labels."""

# %%
import subprocess
print('OK')


# %%
import os
from pathlib import Path


MODEL_IN   = '/kaggle/input/datasets/bogdanuki/t5-nl2dsl-final/t5-nl2dsl-final'
PSEUDO_TSV = '/kaggle/input/datasets/bogdanuki/pseudolabels/pseudolabels.tsv'
MODEL_OUT  = '/kaggle/working/t5-nl2dsl-humans'

print('Model postoji:      ', Path(MODEL_IN).exists())
print('Pseudolabels postoje:', Path(PSEUDO_TSV).exists())


# %%
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

MAX_LEN_IN  = 64
MAX_LEN_OUT = 128
BATCH       = 16
LR          = 5e-5
EPOCHS      = 5
EVAL_STEPS  = 100

rows = []
with open(PSEUDO_TSV, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('question'):
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            rows.append({'question': parts[0], 'program': parts[1]})

random.seed(42)
random.shuffle(rows)
split = int(len(rows) * 0.9)
train_rows, val_rows = rows[:split], rows[split:]
print(f'Train: {len(train_rows)}  Val: {len(val_rows)}')

tokenizer = T5Tokenizer.from_pretrained(MODEL_IN)

class DSLDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = tokenizer(row['question'], max_length=MAX_LEN_IN,
                        truncation=True, padding='max_length', return_tensors='pt')
        dec = tokenizer(row['program'],  max_length=MAX_LEN_OUT,
                        truncation=True, padding='max_length', return_tensors='pt')
        labels = dec.input_ids.squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            'input_ids':      enc.input_ids.squeeze(),
            'attention_mask': enc.attention_mask.squeeze(),
            'labels':         labels,
        }

train_ds = DSLDataset(train_rows)
val_ds   = DSLDataset(val_rows)

model = T5ForConditionalGeneration.from_pretrained(MODEL_IN)

args = TrainingArguments(
    output_dir                   = MODEL_OUT,
    num_train_epochs             = EPOCHS,
    per_device_train_batch_size  = BATCH,
    per_device_eval_batch_size   = BATCH,
    learning_rate                = LR,
    lr_scheduler_type            = 'cosine',
    warmup_steps                 = 50,
    eval_strategy                = 'steps',
    eval_steps                   = EVAL_STEPS,
    save_steps                   = EVAL_STEPS,
    load_best_model_at_end       = True,
    metric_for_best_model        = 'eval_loss',
    fp16                         = False,
    logging_steps                = 50,
    report_to                    = 'none',
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
print(f'Model sacuvan: {MODEL_OUT}')
print('Skini ceo t5-nl2dsl-humans/ folder iz Output panela.')


# %%
import json
import torch

print('\n══ Training History ════════════════════════════════════')
train_logs = []
eval_logs  = []
for entry in trainer.state.log_history:
    if 'loss' in entry and 'eval_loss' not in entry:
        train_logs.append({'step': entry.get('step'), 'loss': round(entry['loss'], 4),
                           'epoch': round(entry.get('epoch', 0), 2)})
    if 'eval_loss' in entry:
        eval_logs.append({'step': entry.get('step'), 'eval_loss': round(entry['eval_loss'], 4),
                          'epoch': round(entry.get('epoch', 0), 2)})

print(f'{"Epoch":>6}  {"Step":>6}  {"Train Loss":>11}  {"Val Loss":>10}')
eval_by_step = {e['step']: e['eval_loss'] for e in eval_logs}
for t in train_logs:
    vl = eval_by_step.get(t['step'], '')
    vl_str = f'{vl:.4f}' if vl != '' else '      —'
    print(f"{t['epoch']:>6.2f}  {t['step']:>6}  {t['loss']:>11.4f}  {vl_str:>10}")

print('\n── Per-epoch val loss ──')
from collections import defaultdict
epoch_eval = defaultdict(list)
for e in eval_logs:
    epoch_eval[int(e['epoch'])].append(e['eval_loss'])
for ep in sorted(epoch_eval):
    avg = sum(epoch_eval[ep]) / len(epoch_eval[ep])
    print(f'  Epoch {ep}: val_loss = {avg:.4f}')

print('\n══ Exact Match Accuracy (val set) ══════════════════════')
model.eval()
SAMPLE = min(300, len(val_rows))
val_sample = val_rows[:SAMPLE]

correct = 0
predictions_log = []
for row in val_sample:
    inputs = tokenizer(
        row['question'], return_tensors='pt',
        max_length=MAX_LEN_IN, truncation=True
    ).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=MAX_LEN_OUT, num_beams=4)
    pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    gt   = row['program'].strip()
    match = (pred == gt)
    correct += int(match)
    predictions_log.append({'question': row['question'], 'gt': gt, 'pred': pred, 'match': match})

exact_match_pct = correct / SAMPLE * 100
print(f'Exact match: {correct}/{SAMPLE} = {exact_match_pct:.1f}%')
print('(Napomena: exact match na pseudo-labelima — ne na stvarnim CLEVR-Humans odgovorima)')

wrong = [p for p in predictions_log if not p['match']]
print(f'\nPrvih 5 pogresnih:')
for p in wrong[:5]:
    print(f'  Q: {p["question"][:70]}')
    print(f'  GT:   {p["gt"][:80]}')
    print(f'  PRED: {p["pred"][:80]}')
    print()

summary = {
    'config': {
        'base_model': MODEL_IN,
        'train_samples': len(train_rows),
        'val_samples': len(val_rows),
        'epochs': EPOCHS,
        'lr': LR,
        'batch_size': BATCH,
        'fp16': args.fp16,
    },
    'training_history': {
        'train_logs': train_logs,
        'eval_logs': eval_logs,
    },
    'val_exact_match': {
        'sample_size': SAMPLE,
        'correct': correct,
        'accuracy_pct': round(exact_match_pct, 2),
    },
    'sample_predictions': predictions_log[:20],
}

results_path = f'{MODEL_OUT}/training_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\n✓ Rezultati sačuvani: {results_path}')
print('Skini t5-nl2dsl-humans/ folder — sadrži model + training_results.json')
