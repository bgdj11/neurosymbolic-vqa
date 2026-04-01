import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ZERO_INPUT_OPS = {'scene'}


def program_to_linear(program: List[Dict]) -> str:
    tokens = []
    for i, step in enumerate(program):
        func = step['function']
        value_inputs = step.get('value_inputs', [])
        side_inputs = step.get('side_inputs', [])
        inputs = step.get('inputs', [])

        val = (value_inputs + side_inputs)
        val_str = (' ' + val[0]) if val else ''

        if len(inputs) == 2:
            # Binary op: always explicit refs
            token = f'{func}:{inputs[0]},{inputs[1]}{val_str}'
        elif len(inputs) == 1 and inputs[0] != i - 1:
            # Single input that is NOT the immediately preceding step: explicit ref
            token = f'{func}:{inputs[0]}{val_str}'
        else:
            # inputs=[] (scene) or inputs=[i-1] (sequential): implicit
            token = f'{func}{val_str}'

        tokens.append(token)

    return ' | '.join(tokens)


def linear_to_program(linear: str) -> List[Dict]:
    tokens = [t.strip() for t in linear.split('|')]
    program = []
    prev = -1

    for token in tokens:
        if not token:
            continue

        first_word = token.split(' ')[0]
        if ':' in first_word:
            head, refs_str = first_word.split(':', 1)
            ref_parts = refs_str.split(',')
            try:
                inputs = [int(r) for r in ref_parts]
            except ValueError:
                inputs = []
            rest = token[len(first_word):].strip()
        else:
            head = first_word
            inputs = None  # resolved below
            rest = token[len(head):].strip()

        func = head
        value_inputs = [rest] if rest else []

        # Resolve implicit inputs
        if inputs is None:
            if func in _ZERO_INPUT_OPS:
                inputs = []
            else:
                inputs = [prev] if prev >= 0 else []

        step = {
            'function': func,
            'inputs': inputs,
            'value_inputs': value_inputs,
            'side_inputs': [],
        }
        program.append(step)
        prev = len(program) - 1

    return program


def _programs_equal(a: List[Dict], b: List[Dict]) -> bool:
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa['function'] != sb['function']:
            return False
        if sa['inputs'] != sb['inputs']:
            return False
        va = sa.get('value_inputs', []) + sa.get('side_inputs', [])
        vb = sb.get('value_inputs', []) + sb.get('side_inputs', [])
        if va != vb:
            return False
    return True


def verify_roundtrip(program: List[Dict]) -> Tuple[bool, Optional[str]]:
    linear = program_to_linear(program)
    recovered = linear_to_program(linear)
    ok = _programs_equal(program, recovered)
    if not ok:
        return False, linear
    return True, linear


def prepare_clevr_split(
    jsonl_path: str,
    output_tsv: str,
    split_name: str,
    roundtrip_check: bool = True,
) -> Dict:
    jsonl_path = Path(jsonl_path)
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'written': 0,
        'roundtrip_fail': 0,
        'missing_program': 0,
    }

    with open(jsonl_path, encoding='utf-8') as f_in, \
         open(output_tsv, 'w', newline='', encoding='utf-8') as f_out:

        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(['question', 'program'])

        for line in f_in:
            sample = json.loads(line)
            stats['total'] += 1

            program = sample.get('program')
            if not program:
                stats['missing_program'] += 1
                continue

            question = sample['question']

            if roundtrip_check:
                ok, linear = verify_roundtrip(program)
                if not ok:
                    stats['roundtrip_fail'] += 1
                    continue
            else:
                linear = program_to_linear(program)

            writer.writerow([question, linear])
            stats['written'] += 1

    return stats


def run_roundtrip_test(jsonl_path: str) -> Dict:
    total = ok = fail = 0
    first_fail = None

    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            program = sample.get('program')
            if not program:
                continue
            total += 1
            success, linear = verify_roundtrip(program)
            if success:
                ok += 1
            else:
                fail += 1
                if first_fail is None:
                    first_fail = {
                        'question_id': sample['question_id'],
                        'question': sample['question'],
                        'original': program,
                        'linear': linear,
                        'recovered': linear_to_program(linear),
                    }

    return {
        'total': total,
        'ok': ok,
        'fail': fail,
        'pass_rate': ok / total if total else 0.0,
        'first_fail': first_fail,
    }


def main():
    base = Path(__file__).resolve().parent.parent
    clevr_dir = base / 'datasets' / 'converted' / 'clevr'
    output_dir = base / 'datasets' / 'nl2dsl'

    print('NL2DSL Data Preparation')
    print()

    for split in ['train', 'val']:
        jsonl = clevr_dir / f'clevr_{split}.jsonl'
        if not jsonl.exists():
            print(f'  SKIP {split}: {jsonl} not found')
            continue

        print(f'[{split}] Round-trip verification...', end=' ', flush=True)
        rt = run_roundtrip_test(str(jsonl))
        print(f'{rt["pass_rate"]:.2%} ({rt["ok"]}/{rt["total"]})')

        if rt['fail'] > 0 and rt['first_fail']:
            ff = rt['first_fail']
            print(f'  First failure: {ff["question_id"]}')
            print(f'  Q: {ff["question"]}')
            print(f'  Linear: {ff["linear"]}')

        print(f'[{split}] Writing TSV...', end=' ', flush=True)
        out = output_dir / f'clevr_{split}.tsv'
        stats = prepare_clevr_split(str(jsonl), str(out), split)
        print(f'done. {stats["written"]}/{stats["total"]} rows -> {out}')
        print()


if __name__ == '__main__':
    main()
