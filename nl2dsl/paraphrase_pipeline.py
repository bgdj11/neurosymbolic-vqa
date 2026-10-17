"""Extract CLEVR questions and assemble a paraphrase training mix."""

import csv, json, random, sys
from pathlib import Path

from nl2dsl.prepare_data import program_to_linear

CLEVR_TRAIN_JSONL = 'datasets/converted/clevr/clevr_train.jsonl'
CLEVR_TRAIN_TSV   = 'datasets/nl2dsl/clevr_train.tsv'
EXTRACT_OUT     = 'datasets/nl2dsl/for_paraphrase.jsonl'
PARAPHRASES_IN  = 'datasets/nl2dsl/paraphrases_llm.jsonl'
TRAIN_OUT       = 'datasets/nl2dsl/paraphrase_mix_train.tsv'
VAL_OUT         = 'datasets/nl2dsl/paraphrase_mix_val.tsv'

N_QUESTIONS  = 40_000
CLEVR_ANCHOR = 40_000
VAL_N        = 2_000

SEED = 42
random.seed(SEED)


def _question_type(program: list) -> str:
    if not program:
        return 'other'
    terminal = program[-1]['function']
    if terminal == 'count':
        return 'count'
    if terminal == 'exist':
        return 'exist'
    if terminal.startswith('query_'):
        return 'query'
    if terminal.startswith('equal_') or terminal in ('greater_than', 'less_than'):
        return 'compare'
    return 'other'


def extract():
    records = []
    with open(CLEVR_TRAIN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if not d['program']:
                continue
            try:
                lin = program_to_linear(d['program'])
            except Exception:
                continue
            records.append({
                'id':       d['question_id'],
                'question': d['question'],
                'type':     _question_type(d['program']),
                '_program': lin,
            })

    random.shuffle(records)
    sample = records[:N_QUESTIONS]

    internal_path = Path(EXTRACT_OUT).with_suffix('.internal.jsonl')
    with open(internal_path, 'w') as f:
        for r in sample:
            f.write(json.dumps(r) + '\n')

    with open(EXTRACT_OUT, 'w') as f:
        for r in sample:
            f.write(json.dumps({'id': r['id'], 'question': r['question'], 'type': r['type']}) + '\n')

    print(f'Ekstrahovano: {len(sample):,} pitanja -> {EXTRACT_OUT}')
    print()
    print('Prompt za LLM (zameni [type] i [question] za svaki red):')
    print('------------------------------------------------------------')
    print('Rewrite this question in a more natural, conversational way,')
    print('as a person would naturally ask it. Keep the meaning identical.')
    print('IMPORTANT: This is a [type] question. The rewrite must remain')
    print('a [type] question — do not change the question type.')
    print('Return only the rewritten question, nothing else.')
    print('Question: "[question]"')
    print('------------------------------------------------------------')
    print(f'Sacuvaj LLM odgovore u: {PARAPHRASES_IN}')
    print(f'Format: {{"id": "...", "paraphrase": "..."}}')


def build():
    internal_path = Path(EXTRACT_OUT).with_suffix('.internal.jsonl')
    id_to_program = {}
    with open(internal_path) as f:
        for line in f:
            r = json.loads(line)
            id_to_program[r['id']] = r['_program']

    paraphrase_pairs = []
    skipped = 0
    with open(PARAPHRASES_IN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prog = id_to_program.get(r['id'])
            if not prog or not r.get('paraphrase', '').strip():
                skipped += 1
                continue
            paraphrase_pairs.append((r['paraphrase'].strip(), prog))

    print(f'Ucitano parafraziranih parova: {len(paraphrase_pairs):,}  (preskoceno: {skipped})')

    random.shuffle(paraphrase_pairs)
    para_val   = paraphrase_pairs[:VAL_N]
    para_train = paraphrase_pairs[VAL_N:]

    anchor = []
    with open(CLEVR_TRAIN_TSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            anchor.append((row['question'], row['program']))
    random.shuffle(anchor)
    anchor = anchor[:CLEVR_ANCHOR]

    train = para_train + anchor
    random.shuffle(train)

    def write_tsv(path, pairs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['question', 'program'])
            for q, p in pairs:
                w.writerow([q, p])

    write_tsv(TRAIN_OUT, train)
    write_tsv(VAL_OUT,   para_val)

    total = len(train)
    print(f'Train — parafraziranih: {len(para_train):,} | CLEVR kotvi: {len(anchor):,} | ukupno: {total:,}')
    print(f'Val   — parafraziranih: {len(para_val):,}')
    print(f'Sacuvano: {TRAIN_OUT}  i  {VAL_OUT}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'extract':
        extract()
    elif cmd == 'build':
        build()
    else:
        print('Upotreba:')
        print('  python -m nl2dsl.paraphrase_pipeline extract')
        print('  python -m nl2dsl.paraphrase_pipeline build')
