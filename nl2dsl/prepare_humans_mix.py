"""Build an exploratory training mix from Humans validation questions."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from nl2dsl.prepare_data import program_to_linear

SEED = 42
CLEVR_TRAIN_TSV = 'datasets/nl2dsl/clevr_train.tsv'
CLEVR_VAL_JSONL = 'datasets/converted/clevr/clevr_val.jsonl'
HUMANS_JSONL    = 'datasets/converted/clevr_humans/clevr_humans_val.jsonl'
OUT_TRAIN       = 'datasets/nl2dsl/humans_mix_train.tsv'
OUT_VAL         = 'datasets/nl2dsl/humans_mix_val.tsv'

CLEVR_ANCHOR    = 500
VAL_HUMANS_N    = 300

random.seed(SEED)


def infer_terminal(question: str) -> str | None:
    q = question.lower().strip()
    if q.startswith('how many') or 'number of' in q:
        return 'count'
    if (q.startswith('is there') or q.startswith('are there')
            or q.startswith('are any') or q.startswith('does')):
        return 'exist'
    if 'what color' in q or 'what is the color' in q:
        return 'query_color'
    if ('what shape' in q or 'what is the shape' in q
            or 'what form' in q or 'what type of shape' in q):
        return 'query_shape'
    if 'what size' in q or 'what is the size' in q or 'how big' in q:
        return 'query_size'
    if ('what material' in q or 'what is the material' in q
            or 'made of' in q or 'rubber or' in q or 'metal or' in q
            or 'matte or' in q or 'shiny or' in q):
        return 'query_material'
    if 'same color' in q:
        return 'equal_color'
    if 'same shape' in q or 'same form' in q:
        return 'equal_shape'
    if 'same size' in q:
        return 'equal_size'
    if 'same material' in q:
        return 'equal_material'
    if 'more' in q and ('than' in q or 'fewer' in q or 'less' in q):
        return 'greater_than'
    return None


def build_val_index():
    """image_id -> list[(answer_str, terminal_op, program_linear)]"""
    idx = defaultdict(list)
    with open(CLEVR_VAL_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if not d['program']:
                continue
            try:
                lin = program_to_linear(d['program'])
            except Exception:
                continue
            terminal = d['program'][-1]['function']
            idx[d['image_id']].append(
                (str(d['answer']).lower(), terminal, lin)
            )
    return idx


def collect_humans_pairs(val_index):
    """
    Uparivanje po tri kriterijuma:
      1. isti image_id
      2. isti odgovor
      3. isti tip programa (terminal operacija) procenjen iz teksta pitanja
    """
    pairs = []
    skipped_no_type = 0
    skipped_no_hit  = 0

    with open(HUMANS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            expected_terminal = infer_terminal(d['question'])
            if expected_terminal is None:
                skipped_no_type += 1
                continue

            answer = str(d['answer']).lower()
            hits = [
                lin for ans, term, lin in val_index.get(d['image_id'], [])
                if ans == answer and term == expected_terminal
            ]
            if not hits:
                skipped_no_hit += 1
                continue

            pairs.append((d['question'], hits[0]))

    print(f'  Tip pitanja neodrediv:       {skipped_no_type}')
    print(f'  Nema podudarajuceg programa: {skipped_no_hit}')
    print(f'  Upareno:                     {len(pairs)}')
    return pairs


def load_clevr_anchor(n):
    rows = []
    with open(CLEVR_TRAIN_TSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append((row['question'], row['program']))
    random.shuffle(rows)
    return rows[:n]


def write_tsv(path, pairs):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['question', 'program'])
        for q, p in pairs:
            writer.writerow([q, p])
    print(f'  Sacuvano: {path}  ({len(pairs):,} redova)')


def main():
    print('Gradim indeks CLEVR-val programa...')
    val_index = build_val_index()

    print('Uparam CLEVR-Humans pitanja (tri kriterijuma)...')
    humans_pairs = collect_humans_pairs(val_index)

    random.shuffle(humans_pairs)
    humans_train = humans_pairs[:-VAL_HUMANS_N] if len(humans_pairs) > VAL_HUMANS_N else humans_pairs
    humans_val   = humans_pairs[-VAL_HUMANS_N:] if len(humans_pairs) > VAL_HUMANS_N else []

    print(f'\nUcitavam {CLEVR_ANCHOR} CLEVR-train kotvi...')
    anchor_rows = load_clevr_anchor(CLEVR_ANCHOR)

    train_pairs = humans_train + anchor_rows
    random.shuffle(train_pairs)

    Path('datasets/nl2dsl').mkdir(parents=True, exist_ok=True)
    print('\nPisem TSV fajlove...')
    write_tsv(OUT_TRAIN, train_pairs)
    write_tsv(OUT_VAL,   humans_val)

    total = len(train_pairs)
    h_pct = len(humans_train) / total * 100
    c_pct = len(anchor_rows)  / total * 100
    print(f'\nMiks: {h_pct:.0f}% CLEVR-Humans + {c_pct:.0f}% CLEVR-train  (ukupno {total:,})')
    print('Gotovo. Uploaduj datasets/nl2dsl/ na Kaggle.')


if __name__ == '__main__':
    main()
