"""Generate question paraphrases through the Anthropic API."""

import json
import os
import time
from pathlib import Path

import anthropic
from tqdm import tqdm

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL     = "claude-haiku-4-5-20251001"
INPUT_FILE  = "datasets/nl2dsl/for_paraphrase.jsonl"
OUTPUT_FILE = "datasets/nl2dsl/paraphrases_llm.jsonl"

DELAY_BETWEEN_CALLS = 0.05

PROMPT_TEMPLATE = (
    "Rewrite this question in a more natural, conversational way, "
    "as a person would naturally ask it. Keep the meaning identical. "
    "Return only the rewritten question, nothing else.\n"
    "Question: {question}"
)

client = None


def already_done(output_path):
    done = set()
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done


def paraphrase(question: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
    )
    return msg.content[0].text.strip()


def main():
    global client
    if not API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY before requesting paraphrases.")
    client = anthropic.Anthropic(api_key=API_KEY)
    questions = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    done = already_done(OUTPUT_FILE)
    todo = [q for q in questions if q["id"] not in done]

    print(f"Ukupno: {len(questions)}  |  Vec obradjeno: {len(done)}  |  Preostalo: {len(todo)}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for item in tqdm(todo, desc="Parafraziranje"):
            try:
                para = paraphrase(item["question"])
                out.write(json.dumps({"id": item["id"], "paraphrase": para}) + "\n")
                out.flush()
                time.sleep(DELAY_BETWEEN_CALLS)
            except Exception as e:
                print(f"\nGreska za {item['id']}: {e} — preskacam")
                time.sleep(1)

    print(f"\nGotovo. Rezultati: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
