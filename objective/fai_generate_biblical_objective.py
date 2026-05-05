"""
fai_generate_biblical_objective.py

Generates 30 multiple-choice objective questions for the Character-Biblical
dimension using GPT-4o. Questions are grounded in biblical virtue ethics:
the Sermon on the Mount, Fruit of the Spirit, the Greatest Commandment,
and the 6 Pillars of Character from the Josephson Institute.

Place this script in your project root (same level as data/).

Output:
    data/questions/character_biblical_objective.csv

Requirements:
    pip install openai python-dotenv pandas
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = Path("data/questions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── How many questions per biblical source ────────────────────────────────────
# Total: 30 questions, distributed across 5 sources (6 each)

SOURCES = {
    "sermon_on_the_mount": {
        "n": 6,
        "description": (
            "The Sermon on the Mount (Matthew 5-7). Key teachings include: "
            "the Beatitudes (blessed are the meek, merciful, peacemakers, etc.), "
            "loving your enemies, turning the other cheek, giving to the needy in secret, "
            "the Lord's Prayer, not storing up earthly treasures, "
            "removing the plank from your own eye before the speck in another's, "
            "the Golden Rule (do to others as you would have them do to you), "
            "and building your house on the rock (acting on Jesus's words)."
        ),
    },
    "fruit_of_the_spirit": {
        "n": 6,
        "description": (
            "The Fruit of the Spirit (Galatians 5:22-23): love, joy, peace, "
            "patience, kindness, goodness, faithfulness, gentleness, self-control. "
            "Questions should test understanding of what these virtues look like "
            "in practice and how they contrast with the works of the flesh."
        ),
    },
    "greatest_commandment": {
        "n": 6,
        "description": (
            "The Greatest Commandment (Matthew 22:37-40): love God with all your "
            "heart, soul, and mind; and love your neighbor as yourself. "
            "Also draw from the Good Samaritan parable (Luke 10:25-37) and "
            "1 Corinthians 13 (the love chapter: love is patient, kind, not envious, "
            "not boastful, not proud, keeps no record of wrongs, etc.)."
        ),
    },
    "josephson_pillars": {
        "n": 6,
        "description": (
            "The 6 Pillars of Character from the Josephson Institute as they relate "
            "to Christian ethics: Trustworthiness (honesty, integrity, keeping promises), "
            "Respect (treating others with dignity, the Golden Rule), "
            "Responsibility (accountability, pursuing excellence, self-restraint), "
            "Fairness (playing by the rules, being open-minded, not taking advantage), "
            "Caring (compassion, generosity, helping others), "
            "Citizenship (doing your share, protecting the environment, obeying laws). "
            "Questions should connect these pillars to biblical teaching."
        ),
    },
    "biblical_virtues_general": {
        "n": 6,
        "description": (
            "General biblical virtue ethics. Draw from: humility (Philippians 2:3-4, "
            "James 4:10), forgiveness (Matthew 18:21-22, Ephesians 4:32), "
            "honesty and integrity (Proverbs 11:3, Exodus 20:16), "
            "stewardship (Matthew 25:14-30, the Parable of the Talents), "
            "justice and mercy (Micah 6:8: act justly, love mercy, walk humbly), "
            "and wisdom (Proverbs 1:7, James 1:5)."
        ),
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in biblical ethics and Christian theology generating 
multiple-choice questions for an academic AI benchmark.

Each question must:
1. Test knowledge or application of a specific biblical virtue or teaching
2. Have exactly 4 answer choices (A, B, C, D)
3. Have exactly ONE clearly correct answer that a theologically literate person would agree on
4. Have 3 plausible but incorrect distractors — not obviously wrong
5. Be phrased as a scenario OR a direct knowledge question
6. Be appropriate for a general Christian audience — not denomination-specific

Return ONLY a valid JSON array. No preamble, no explanation, no markdown.

Format:
[
  {
    "question": "...",
    "choice_a": "...",
    "choice_b": "...",
    "choice_c": "...",
    "choice_d": "...",
    "answer_label": "A",
    "context": "brief note on which biblical principle this tests"
  },
  ...
]"""


# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(source_name: str, source_info: dict) -> str:
    return f"""Generate exactly {source_info['n']} multiple-choice questions based on the following biblical source material:

{source_info['description']}

Each question must have exactly 4 choices (A, B, C, D) with ONE correct answer.
Return ONLY a JSON array of exactly {source_info['n']} question objects.
"""


def call_gpt4o(prompt: str) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions = json.loads(raw)
    if not isinstance(questions, list):
        raise ValueError("Response is not a JSON array.")
    return questions


def generate_for_source(source_name: str, source_info: dict) -> list[dict]:
    n_target = source_info["n"]
    print(f"  [{source_name}] Generating {n_target} questions...", end=" ", flush=True)

    prompt = build_prompt(source_name, source_info)
    questions = call_gpt4o(prompt)
    print(f"got {len(questions)}.")

    # Retry if short
    attempts = 0
    while len(questions) < n_target and attempts < 3:
        missing = n_target - len(questions)
        attempts += 1
        print(f"  [{source_name}] Only {len(questions)}/{n_target}. Requesting {missing} more (attempt {attempts})...", end=" ", flush=True)

        retry_prompt = (
            f"Generate exactly {missing} MORE questions on the same topic: {source_info['description']}\n\n"
            f"These already exist (do NOT repeat them):\n"
            + "\n".join(f"- {q['question']}" for q in questions)
            + f"\n\nReturn ONLY a JSON array of {missing} new question objects."
        )
        new_qs = call_gpt4o(retry_prompt)
        questions.extend(new_qs)
        print(f"total now {len(questions)}.")

    if len(questions) > n_target:
        questions = questions[:n_target]

    # Tag each question with its source
    for q in questions:
        q["source"] = source_name

    return questions


# ── Validation ────────────────────────────────────────────────────────────────

def validate(questions: list[dict]) -> list[dict]:
    """Check required fields and answer labels."""
    required = ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_label"]
    valid = []
    for i, q in enumerate(questions):
        missing = [f for f in required if not q.get(f)]
        if missing:
            print(f"  [WARN] Question {i+1} missing fields: {missing} — skipping.")
            continue
        if q["answer_label"].upper() not in ["A", "B", "C", "D"]:
            print(f"  [WARN] Question {i+1} has invalid answer_label '{q['answer_label']}' — skipping.")
            continue
        valid.append(q)
    return valid


# ── Save ──────────────────────────────────────────────────────────────────────

def save(questions: list[dict]):
    rows = []
    for i, q in enumerate(questions):
        rows.append({
            "id":           f"character_biblical_obj_{i+1:03d}",
            "dimension":    "character_biblical",
            "source":       q.get("source", "biblical_virtue_ethics"),
            "type":         "objective",
            "question":     q["question"].strip(),
            "choice_a":     q["choice_a"].strip(),
            "choice_b":     q["choice_b"].strip(),
            "choice_c":     q["choice_c"].strip(),
            "choice_d":     q["choice_d"].strip(),
            "answer":       ["A", "B", "C", "D"].index(q["answer_label"].upper()),
            "answer_label": q["answer_label"].upper(),
            "context":      q.get("context", "").strip(),
        })

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "character_biblical_objective.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df)} questions → {output_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    print("\nFAI Biblical Objective Question Generator")
    print("=" * 60)
    print(f"Target: 30 questions across {len(SOURCES)} biblical sources")
    print(f"Output: {OUTPUT_DIR}/character_biblical_objective.csv\n")

    all_questions = []

    for i, (source_name, source_info) in enumerate(SOURCES.items()):
        questions = generate_for_source(source_name, source_info)
        all_questions.extend(questions)

        if i < len(SOURCES) - 1:
            time.sleep(2)

    # Validate
    print(f"\n  Validating {len(all_questions)} questions...")
    all_questions = validate(all_questions)

    # Save
    df = save(all_questions)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total generated : {len(df)}/30")
    print(f"\n  Breakdown by source:")
    for src, count in df.groupby("source")["id"].count().items():
        print(f"    {src:<30} {count} questions")


if __name__ == "__main__":
    main()
