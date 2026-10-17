"""Generate candidate programs and retain those matching known answers."""

import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL             = "claude-haiku-4-5-20251001"

HUMANS_JSONL   = "datasets/converted/clevr_humans/clevr_humans_pseudo_pool.jsonl"
OUTPUT_DIR     = "results/llm_pseudolabels"

N_CANDIDATES   = 5
TEMPERATURE    = 0.4
MAX_TOKENS     = 512
REQUEST_DELAY  = 0.3
LIMIT          = None

import os, sys, json, time, re
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsl.executor import Executor, ExecutionError
from dsl.scene_graph import SceneGraph
from nl2dsl.prepare_data import linear_to_program
from nl2dsl.llm_translator import FEW_SHOT_EXAMPLES

os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT    = f"{OUTPUT_DIR}/checkpoint.jsonl"
PSEUDOLABELS  = f"{OUTPUT_DIR}/pseudolabels.tsv"
SUMMARY_FILE  = f"{OUTPUT_DIR}/summary.json"

_MULTI_SYSTEM = """You are a semantic parser. Convert natural language questions about 3D scenes into a DSL (Domain Specific Language).

DSL rules:
- Programs are sequences of operations separated by " | "
- Steps are numbered 0, 1, 2, ... from left to right
- scene: returns all objects (always first)
- filter_color <color>: Colors: red, blue, green, yellow, purple, gray, brown, cyan
- filter_shape <shape>: Shapes: cube, sphere, cylinder. "block"=cube
- filter_size <size>: Sizes: large, small. "big"=large
- filter_material <mat>: Materials: metal, rubber. "shiny"=metal, "matte"=rubber
- unique: select the single object
- count: count objects → integer
- exist: do any objects match → yes/no
- query_color / query_shape / query_size / query_material: get attribute of unique object
- same_shape / same_size / same_color / same_material: OTHER objects with same attribute
- relate <dir>: objects spatially related. Directions: left, right, front, behind
- greater_than, less_than, equal_integer: compare two integers (use :i,j refs)
- equal_color, equal_shape, equal_size, equal_material: compare two attribute values (use :i,j refs)
- union:i,j / intersect:i,j: set operations

Step reference syntax:
- Use :i ONLY to refer to step number i (0-indexed)
- Use op:i when input is NOT the immediately previous step (branching)
- Use op:i,j for binary operations taking two step outputs

OUTPUT FORMAT — return EXACTLY {n} lines, nothing else:
PROG_1: <dsl program>
PROG_2: <dsl program>
...
PROG_{n}: <dsl program>

If you are confident in one answer, repeat it with minor variations in filter order or approach.
Each line must start with "scene".
"""

def _build_multi_prompt(question: str, n: int) -> tuple[str, str]:
    examples_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in FEW_SHOT_EXAMPLES)
    system = _MULTI_SYSTEM.replace("{n}", str(n)).replace("PROG_{n}", f"PROG_{n}")
    user   = f"Examples:\n{examples_text}\n\nQ: {question}"
    return system, user


def call_anthropic(question: str, n: int, retries: int = 3) -> Optional[str]:
    import urllib.request, urllib.error

    system, user = _build_multi_prompt(question, n)
    payload = json.dumps({
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system":     system,
        "messages":   [{"role": "user", "content": user}],
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limit — cekam {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {e.code}: {body[:200]}")
                break
        except Exception as e:
            print(f"    API greska (pokusaj {attempt+1}): {e}")
            time.sleep(2)

    return None


def parse_candidates(response: str) -> list[str]:
    programs = []
    for line in response.strip().splitlines():
        m = re.match(r"PROG_\d+:\s*(.+)", line.strip())
        if m:
            prog = m.group(1).strip()
            if prog.startswith("scene"):
                programs.append(prog)
    return programs


def _question_type(question: str) -> str:
    q = question.lower()
    if q.startswith("how many") or "how many" in q:
        return "count"
    if any(q.startswith(w) for w in ("is there", "are there", "is the", "are the",
                                      "does ", "do the", "is it")):
        return "boolean"
    if q.startswith(("what ", "which ")):
        return "attribute"
    return "unknown"

_EXPECTED_FINALS = {
    "count":     {"count", "greater_than", "less_than", "equal_integer"},
    "boolean":   {"exist", "equal_color", "equal_shape", "equal_size",
                  "equal_material", "greater_than", "less_than", "equal_integer"},
    "attribute": {"query_color", "query_shape", "query_size", "query_material"},
    "unknown":   set(),
}

def _type_ok(program: list, q_type: str) -> bool:
    if not program or q_type == "unknown":
        return True
    last = program[-1]["function"].split(":")[0]
    allowed = _EXPECTED_FINALS[q_type]
    return not allowed or last in allowed


def validate(prog_str: str, scene_graph: dict,
             expected_answer: str, q_type: str) -> tuple[bool, str]:
    """
    Vraca (is_correct, status_string).
    """
    try:
        prog = linear_to_program(prog_str)
    except Exception as e:
        return False, f"parse_error: {e}"

    if not _type_ok(prog, q_type):
        return False, f"type_mismatch: ends with {prog[-1]['function']}"

    try:
        ans = str(Executor(SceneGraph(scene_graph)).execute(prog)).lower()
    except ExecutionError as e:
        return False, f"exec_error: {e}"
    except Exception as e:
        return False, f"runtime_error: {e}"

    if ans == str(expected_answer).lower():
        return True, "correct"
    return False, f"wrong: got={ans} expected={expected_answer}"


def main():
    if not ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY before running pseudo-label generation.")
    samples = []
    with open(HUMANS_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if LIMIT and i >= LIMIT:
                break
            samples.append(json.loads(line))
    print(f"Ukupno pitanja: {len(samples):,}")

    done: dict[str, dict] = {}
    if Path(CHECKPOINT).exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["question_id"]] = r
        print(f"Checkpoint: {len(done)} vec obradjeno.")

    todo = [s for s in samples if s["question_id"] not in done]
    print(f"Preostalo: {len(todo)}")
    if not todo:
        print("Sve je vec obradjeno.")
        _export(done, samples)
        return

    ckpt_f = open(CHECKPOINT, "a", encoding="utf-8")
    counters = {"total": 0, "hit": 0, "api_err": 0}
    t0 = time.time()

    for i, sample in enumerate(todo):
        qid   = sample["question_id"]
        q     = sample["question"]
        ans   = sample["answer"]
        sg    = sample["scene_graph"]
        qtype = _question_type(q)

        response = call_anthropic(q, N_CANDIDATES)
        counters["total"] += 1

        if response is None:
            counters["api_err"] += 1
            r = {
                "question_id":    qid,
                "question":        q,
                "answer":          ans,
                "question_type":   qtype,
                "valid_programs":  [],
                "candidates":      [],
                "error":           "api_error",
            }
        else:
            raw_candidates = parse_candidates(response)
            valid_programs = []
            candidates_log = []

            for prog_str in raw_candidates:
                ok, status = validate(prog_str, sg, ans, qtype)
                candidates_log.append({"program": prog_str, "status": status})
                if ok and prog_str not in valid_programs:
                    valid_programs.append(prog_str)

            if valid_programs:
                counters["hit"] += 1

            r = {
                "question_id":    qid,
                "question":        q,
                "answer":          ans,
                "question_type":   qtype,
                "valid_programs":  valid_programs,
                "candidates":      candidates_log,
                "error":           None,
            }

        done[qid] = r
        ckpt_f.write(json.dumps(r) + "\n")
        ckpt_f.flush()

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(todo) - i - 1) / max(rate, 0.01)
            hit_r   = counters["hit"] / max(counters["total"], 1)
            print(
                f"  [{i+1:>4}/{len(todo)}] "
                f"hit={hit_r:.1%}  "
                f"elapsed={elapsed:.0f}s  "
                f"ETA={eta:.0f}s"
            )

        time.sleep(REQUEST_DELAY)

    ckpt_f.close()
    _export(done, samples)


def _export(done: dict, samples: list):
    results = [done[s["question_id"]] for s in samples if s["question_id"] in done]

    total       = len(results)
    hit         = sum(1 for r in results if r.get("valid_programs"))
    api_errs    = sum(1 for r in results if r.get("error") == "api_error")
    total_progs = sum(len(r.get("valid_programs", [])) for r in results)

    by_type: dict[str, list[bool]] = {}
    for r in results:
        t = r.get("question_type", "unknown")
        by_type.setdefault(t, []).append(bool(r.get("valid_programs")))

    print(f"\n{'='*50}")
    print(f"UKUPNO PITANJA:           {total:,}")
    print(f"Sa validnim programom:    {hit:,} ({hit/max(total,1):.1%})  ← LLM accuracy")
    print(f"API greske:               {api_errs}")
    print(f"Ukupno pseudo-labela:     {total_progs:,}")
    print(f"Prosek programa po hitu:  {total_progs/max(hit,1):.1f}")
    print(f"\nPo tipu pitanja:")
    for t, vals in sorted(by_type.items()):
        print(f"  {t:10}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}")
    print(f"{'='*50}")

    with open(PSEUDOLABELS, "w", encoding="utf-8") as f:
        f.write("question\tprogram\n")
        for r in results:
            for prog in r.get("valid_programs", []):
                f.write(f"{r['question']}\t{prog}\n")
    print(f"\nPseudo-labeli: {PSEUDOLABELS}  ({total_progs:,} redova)")

    summary = {
        "total_questions":   total,
        "questions_with_valid_program": hit,
        "llm_accuracy":      round(hit / max(total, 1), 4),
        "total_pseudolabels": total_progs,
        "api_errors":        api_errs,
        "accuracy_by_type":  {t: round(sum(v)/len(v), 4) for t, v in by_type.items()},
        "n_candidates":      N_CANDIDATES,
        "temperature":       TEMPERATURE,
        "model":             MODEL,
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
