"""Compare Humans checkpoints using the original standalone DSL variant."""

# %%
import subprocess
print('OK')


# %%

from typing import List, Dict, Any, Optional


class SceneGraph:
    def __init__(self, scene_data: Dict[str, Any]):
        self.objects = scene_data.get('objects', [])
        self.relationships = scene_data.get('relationships', {})
        self.num_objects = len(self.objects)

    def get_object(self, idx):
        return self.objects[idx]

    def get_attribute(self, idx, attr):
        obj = self.objects[idx]
        if attr not in obj:
            raise KeyError(f"Attr '{attr}' not in object {idx}: {list(obj.keys())}")
        return obj[attr]

    def get_related(self, idx, relation):
        if relation not in self.relationships:
            raise KeyError(f"Relation '{relation}' not found. Available: {list(self.relationships.keys())}")
        rels = self.relationships[relation]
        if idx >= len(rels):
            return []
        return rels[idx]

    def all_objects(self):
        return list(range(self.num_objects))

    def __len__(self):
        return self.num_objects


def _scene(sg, *a):        return sg.all_objects()
def _filter_color(sg, o, c):    return [i for i in o if sg.get_attribute(i,'color')==c]
def _filter_shape(sg, o, s):    return [i for i in o if sg.get_attribute(i,'shape')==s]
def _filter_size(sg, o, s):     return [i for i in o if sg.get_attribute(i,'size')==s]
def _filter_material(sg, o, m): return [i for i in o if sg.get_attribute(i,'material')==m]

def _relate(sg, o, rel):
    res = set()
    for idx in o:
        res.update(sg.get_related(idx, rel))
    return list(res)

def _intersect(sg, a, b): return list(set(a) & set(b))
def _union(sg, a, b):     return list(set(a) | set(b))

def _unique(sg, o):
    if len(o) == 1: return o
    if len(o) == 0: raise ValueError("unique() got 0 objects")
    def area(i):
        b = sg.get_object(i).get('bbox', [0,0,0,0])
        return (b[2]-b[0])*(b[3]-b[1])
    return [max(o, key=area)]

def _query_color(sg, o):    return sg.get_attribute(o[0],'color')    if len(o)==1 else (_ for _ in ()).throw(ValueError(f"query_color got {len(o)}"))
def _query_shape(sg, o):    return sg.get_attribute(o[0],'shape')    if len(o)==1 else (_ for _ in ()).throw(ValueError(f"query_shape got {len(o)}"))
def _query_size(sg, o):     return sg.get_attribute(o[0],'size')     if len(o)==1 else (_ for _ in ()).throw(ValueError(f"query_size got {len(o)}"))
def _query_material(sg, o): return sg.get_attribute(o[0],'material') if len(o)==1 else (_ for _ in ()).throw(ValueError(f"query_material got {len(o)}"))

def _count(sg, o):    return len(o)
def _exist(sg, o):    return len(o) > 0
def _equal_color(sg, a, b):    return a == b
def _equal_shape(sg, a, b):    return a == b
def _equal_size(sg, a, b):     return a == b
def _equal_material(sg, a, b): return a == b
def _equal_integer(sg, a, b):  return a == b
def _less_than(sg, a, b):      return a < b
def _greater_than(sg, a, b):   return a > b

def _same_attr(sg, o, attr):
    if len(o) != 1: raise ValueError(f"same_{attr} expects 1 obj")
    ref = o[0]; val = sg.get_attribute(ref, attr)
    return [i for i in sg.all_objects() if i != ref and sg.get_attribute(i, attr) == val]

def _same_color(sg, o):    return _same_attr(sg, o, 'color')
def _same_shape(sg, o):    return _same_attr(sg, o, 'shape')
def _same_size(sg, o):     return _same_attr(sg, o, 'size')
def _same_material(sg, o): return _same_attr(sg, o, 'material')

OPERATORS = {
    'scene': _scene, 'filter_color': _filter_color, 'filter_shape': _filter_shape,
    'filter_size': _filter_size, 'filter_material': _filter_material,
    'relate': _relate, 'intersect': _intersect, 'union': _union, 'unique': _unique,
    'query_color': _query_color, 'query_shape': _query_shape,
    'query_size': _query_size, 'query_material': _query_material,
    'count': _count, 'exist': _exist,
    'equal_color': _equal_color, 'equal_shape': _equal_shape,
    'equal_size': _equal_size, 'equal_material': _equal_material,
    'equal_integer': _equal_integer, 'less_than': _less_than, 'greater_than': _greater_than,
    'same_color': _same_color, 'same_shape': _same_shape,
    'same_size': _same_size, 'same_material': _same_material,
}


class ExecutionError(Exception):
    pass

def execute_program(scene_data: dict, program: List[Dict]) -> Any:
    sg = SceneGraph(scene_data)
    results = []
    for i, step in enumerate(program):
        func = step.get('function')
        if func not in OPERATORS:
            raise ExecutionError(f"Unknown op: {func}")
        inputs = [results[j] for j in step.get('inputs', [])]
        vals = step.get('value_inputs', []) or step.get('side_inputs', [])
        args = [sg] + inputs + vals
        try:
            results.append(OPERATORS[func](*args))
        except Exception as e:
            raise ExecutionError(f"Step {i} ({func}): {e}") from e
    r = results[-1]
    return ('yes' if r else 'no') if isinstance(r, bool) else r


_ZERO_INPUT_OPS = {'scene'}

def linear_to_program(linear: str) -> List[Dict]:
    tokens = [t.strip() for t in linear.split('|')]
    program = []
    prev = -1
    for token in tokens:
        if not token:
            continue
        first = token.split(' ')[0]
        if ':' in first:
            head, refs_str = first.split(':', 1)
            try:
                inputs = [int(r) for r in refs_str.split(',')]
            except ValueError:
                inputs = []
            rest = token[len(first):].strip()
        else:
            head = first
            inputs = None
            rest = token[len(head):].strip()
        if inputs is None:
            inputs = [] if head in _ZERO_INPUT_OPS else ([prev] if prev >= 0 else [])
        program.append({'function': head, 'inputs': inputs,
                        'value_inputs': [rest] if rest else [], 'side_inputs': []})
        prev = len(program) - 1
    return program


import re

_PATTERNS = [
    r'\bfurthest\b', r'\bfarthest\b', r'\bclosest\b', r'\bnearest\b',
    r'\bleftmost\b', r'\brightmost\b', r'\bmost common\b', r'\bmost frequent\b',
    r'\bfewest\b', r'\bmost\b',
    r'\bif .{1,30} (were|was|are) (added|removed|taken away)\b',
    r'\bwould (there|it|they) be\b', r'\bhow many times\b',
    r'\bhow many .{1,30} (more|fewer|less) than\b', r'\bhow many more .{1,40} than\b',
    r'\bhow much\b',
    r'\bin (the )?middle\b', r'\bbetween .{1,30} and\b', r'\bbetween the\b',
    r'\bhiding\b', r'\bblocked\b', r'\bhidden\b', r'\bobscured\b',
    r'\bwhich (is|one is|are) (larger|smaller|bigger|closer|further|taller|shorter)\b',
    r'\bmost (often|commonly|frequently|represented)\b',
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

def is_dsl_expressible(question: str) -> bool:
    for pat in _COMPILED:
        if pat.search(question):
            return False
    return True


print('DSL engine ucitan.')


# %%
import os
from pathlib import Path

HUMANS_JSONL  = '/kaggle/input/datasets/bogdanuki/clevr-humans/clevr_humans_val.jsonl'
MODEL_BEFORE  = '/kaggle/input/datasets/bogdanuki/t5-nl2dsl-final/t5-nl2dsl-final'
MODEL_AFTER   = '/kaggle/input/datasets/bogdanuki/t5-nl2dsl-humans/t5-nl2dsl-humans'
OUT_DIR       = '/kaggle/working/humans_eval'

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print('humans_val postoji:', Path(HUMANS_JSONL).exists())
print('model_before postoji:', Path(MODEL_BEFORE).exists())
print('model_after postoji:', Path(MODEL_AFTER).exists())


# %%
import json

samples = []
with open(HUMANS_JSONL, encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line.strip()))

print(f'Ukupno pitanja: {len(samples)}')
print(f'Primer: {samples[0]["question"]} -> {samples[0]["answer"]}')


# %%
from collections import Counter, defaultdict

covered   = [s for s in samples if is_dsl_expressible(s['question'])]
uncovered = [s for s in samples if not is_dsl_expressible(s['question'])]

print(f'\n{"="*56}')
print(f'DSL COVERAGE ANALIZA — {len(samples)} pitanja')
print(f'{"="*56}')
print(f'  Expressible:     {len(covered)}/{len(samples)} = {len(covered)/len(samples):.1%}')
print(f'  Not expressible: {len(uncovered)}/{len(samples)} = {len(uncovered)/len(samples):.1%}')

print(f'\nPrimeri pitanja koja DSL ne moze izraziti:')
for s in uncovered[:5]:
    print(f'  "{s["question"]}"')


# %%
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

NUM_BEAMS  = 4
MAX_IN     = 64
MAX_OUT    = 128
BATCH_SIZE = 32

def load_model(path: str):
    tok = T5Tokenizer.from_pretrained(path)
    mdl = T5ForConditionalGeneration.from_pretrained(path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdl.to(device)
    mdl.eval()
    print(f'Model ucitan: {path}  [{device}]')
    return tok, mdl, device

def predict_programs(questions, tok, mdl, device):
    programs = []
    total = len(questions)
    for start in range(0, total, BATCH_SIZE):
        batch_q = questions[start:start+BATCH_SIZE]
        texts = [f'translate to DSL: {q}' for q in batch_q]
        enc = tok(texts, return_tensors='pt', max_length=MAX_IN,
                  truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = mdl.generate(**enc, num_beams=NUM_BEAMS,
                               num_return_sequences=NUM_BEAMS,
                               max_length=MAX_OUT, early_stopping=True)
        for j in range(len(batch_q)):
            beams = out[j*NUM_BEAMS:(j+1)*NUM_BEAMS]
            prog = None
            for beam in beams:
                linear = tok.decode(beam, skip_special_tokens=True).strip()
                try:
                    prog = linear_to_program(linear)
                    if prog and prog[0]['function'] in OPERATORS:
                        break
                except Exception:
                    prog = None
            programs.append(prog)
        if (start // BATCH_SIZE) % 5 == 0:
            print(f'  {min(start+BATCH_SIZE, total)}/{total} pitanja prevedeno...')
    return programs

def run_eval(samples_list, tok, mdl, device, label=''):
    questions = [s['question'] for s in samples_list]
    print(f'\nPrevodim {len(questions)} pitanja [{label}]...')
    programs  = predict_programs(questions, tok, mdl, device)

    results = []
    for s, prog in zip(samples_list, programs):
        expressible = is_dsl_expressible(s['question'])
        sg_data = s.get('scene_graph')
        correct = False
        exec_err = None
        prog_status = 'failed'

        if prog is None:
            prog_status = 'failed'
        elif sg_data is None:
            exec_err = 'no_scene_graph'
            prog_status = 'valid'
        else:
            prog_status = 'valid'
            try:
                result = execute_program(sg_data, prog)
                predicted = str(result).lower()
                correct = (predicted == str(s['answer']).lower())
            except ExecutionError as e:
                exec_err = str(e)

        results.append({
            'question_id': s.get('question_id', ''),
            'question': s['question'],
            'answer': str(s['answer']),
            'correct': correct,
            'prog_status': prog_status,
            'exec_error': exec_err,
            'dsl_expressible': expressible,
        })

    return results

def print_summary(results, label):
    total      = len(results)
    correct    = sum(r['correct'] for r in results)
    failed_nl  = sum(r['prog_status'] == 'failed' for r in results)
    exec_errs  = sum(r['exec_error'] is not None for r in results)
    covered_r  = [r for r in results if r['dsl_expressible']]
    covered_c  = sum(r['correct'] for r in covered_r)

    print(f'\n{"="*60}')
    print(f'REZULTATI — {label}')
    print(f'{"="*60}')
    print(f'  Ukupno pitanja:     {total}')
    print(f'  Tacno:              {correct}/{total} = {correct/total:.1%}')
    print(f'  DSL covered subset: {covered_c}/{len(covered_r)} = {covered_c/len(covered_r):.1%}  ({len(covered_r)/total:.1%} pitanja)')
    print(f'  Failed NL2DSL:      {failed_nl}/{total} = {failed_nl/total:.1%}')
    print(f'  Execution errors:   {exec_errs}/{total} = {exec_errs/total:.1%}')

    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        q = r['question'].lower()
        if q.startswith('how many'):   t = 'count'
        elif q.startswith('is there') or q.startswith('are there'): t = 'exist'
        elif q.startswith('what') or q.startswith('which'): t = 'query'
        else: t = 'other'
        by_type[t][0] += r['correct']
        by_type[t][1] += 1
    print(f'\n  Po tipu pitanja:')
    for t, (c, n) in sorted(by_type.items()):
        print(f'    {t:<8} {c:>4}/{n:<4} = {c/n:.1%}')

    return {
        'label': label,
        'total': total,
        'correct': correct,
        'accuracy': round(correct/total, 4),
        'covered_correct': covered_c,
        'covered_total': len(covered_r),
        'covered_accuracy': round(covered_c/len(covered_r), 4) if covered_r else 0,
        'failed_nl2dsl': failed_nl,
        'exec_errors': exec_errs,
        'by_type': {t: round(c/n, 4) for t, (c,n) in by_type.items()},
    }


# %%
tok_before, mdl_before, device = load_model(MODEL_BEFORE)

results_before = run_eval(samples, tok_before, mdl_before, device, label='BEFORE (t5-nl2dsl-final)')
summary_before = print_summary(results_before, 'BEFORE fine-tune')

del mdl_before, tok_before
if device == 'cuda':
    torch.cuda.empty_cache()


# %%
tok_after, mdl_after, device = load_model(MODEL_AFTER)

results_after = run_eval(samples, tok_after, mdl_after, device, label='AFTER (t5-nl2dsl-humans)')
summary_after = print_summary(results_after, 'AFTER fine-tune')

del mdl_after, tok_after
if device == 'cuda':
    torch.cuda.empty_cache()


# %%
print(f'\n{"="*60}')
print(f'POREĐENJE PRE / POSLE')
print(f'{"="*60}')

def delta(a, b):
    d = b - a
    return f'+{d:.1%}' if d >= 0 else f'{d:.1%}'

print(f'  {"Metrika":<28} {"BEFORE":>8}  {"AFTER":>8}  {"Delta":>8}')
print(f'  {"-"*54}')
print(f'  {"Accuracy (sve)":<28} {summary_before["accuracy"]:.1%}  {summary_after["accuracy"]:.1%}  {delta(summary_before["accuracy"], summary_after["accuracy"])}')
print(f'  {"Accuracy (DSL-covered)":<28} {summary_before["covered_accuracy"]:.1%}  {summary_after["covered_accuracy"]:.1%}  {delta(summary_before["covered_accuracy"], summary_after["covered_accuracy"])}')
print(f'  {"Failed NL2DSL":<28} {summary_before["failed_nl2dsl"]/summary_before["total"]:.1%}  {summary_after["failed_nl2dsl"]/summary_after["total"]:.1%}  {delta(summary_before["failed_nl2dsl"]/summary_before["total"], summary_after["failed_nl2dsl"]/summary_after["total"])}')
print(f'  {"Execution errors":<28} {summary_before["exec_errors"]/summary_before["total"]:.1%}  {summary_after["exec_errors"]/summary_after["total"]:.1%}  {delta(summary_before["exec_errors"]/summary_before["total"], summary_after["exec_errors"]/summary_after["total"])}')
print(f'  {"DSL coverage":<28} {summary_before["covered_total"]/summary_before["total"]:.1%}  {summary_after["covered_total"]/summary_after["total"]:.1%}  (isti)')

output = {
    'dataset': 'clevr_humans_val',
    'n_total': len(samples),
    'n_dsl_covered': len(covered),
    'dsl_coverage_pct': round(len(covered)/len(samples), 4),
    'before': summary_before,
    'after': summary_after,
    'per_question_before': results_before,
    'per_question_after': results_after,
}

out_path = f'{OUT_DIR}/humans_eval_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\nRezultati sacuvani: {out_path}')
print('Skini humans_eval/ folder iz Output panela.')
