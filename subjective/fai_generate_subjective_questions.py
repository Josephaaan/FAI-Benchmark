"""
fai_generate_subjective_questions.py

Generates subjective questions for each FAI dimension using GPT-4o.
Reads a single consolidated grounding .txt file per dimension.
Finance dimension uses a PDF instead of a txt.

Directory structure expected:
    grounding/
    ├── character-Biblical/   character_biblical_subjective_grounding_document.txt
    ├── character-Secular/    character_secular_subjective_grounding_document.txt
    ├── faith/                faith_subjective_grounding_document.txt
    ├── finance/              finance_subjective_grounding_document.pdf
    ├── happiness/            happiness_subjective_grounding_document.txt
    ├── health/               health_subjective_grounding_document.txt
    ├── meaning/              meaning_subjective_grounding_document.txt
    └── relationships/        relationships_subjective_grounding_document.txt

Usage:
    python fai_generate_subjective_questions.py --dimension all
    python fai_generate_subjective_questions.py --dimension faith
    python fai_generate_subjective_questions.py --dimension character_biblical

Output:
    subjective/data/questions/{dimension}_subjective.csv

Requirements:
    pip install openai pypdf python-dotenv
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("Run: pip install pypdf")

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR.parent / "API.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Directory config ──────────────────────────────────────────────────────────
GROUNDING_BASE = BASE_DIR.parent / "grounding"
OUTPUT_DIR = BASE_DIR / "data" / "questions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dimension config ──────────────────────────────────────────────────────────
# grounding_subdir: folder name under GROUNDING_BASE
# grounding_file:   filename inside that folder (txt or pdf)
# use_pdf:          True only for finance
# n_questions:      how many questions to generate

DIMENSION_CONFIG = {
    "character_biblical": {
        "grounding_subdir": "character-Biblical",
        "grounding_file": "character_biblical_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 20,
    },
    "character_secular": {
        "grounding_subdir": "character-Secular",
        "grounding_file": "character_secular_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 20,
    },
    "relationships": {
        "grounding_subdir": "relationships",
        "grounding_file": "relationships_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 40,
    },
    "finance": {
        "grounding_subdir": "finance",
        "grounding_file": "finance_subjective_grounding_document.pdf",
        "use_pdf": True,
        "n_questions": 40,
    },
    "health": {
        "grounding_subdir": "health",
        "grounding_file": "health_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 40,
    },
    "happiness": {
        "grounding_subdir": "happiness",
        "grounding_file": "happiness_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 40,
    },
    "faith": {
        "grounding_subdir": "faith",
        "grounding_file": "faith_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 40,
    },
    "meaning": {
        "grounding_subdir": "meaning",
        "grounding_file": "meaning_subjective_grounding_document.txt",
        "use_pdf": False,
        "n_questions": 40,
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert researcher generating subjective scenario questions 
for an academic AI benchmark evaluating how well AI models support human flourishing.

You will be given a grounding document that contains:
- Source material (scales, frameworks, research findings)
- Generation instructions specifying how many questions to generate and from which sources
- Example questions

Follow the grounding document's instructions precisely.
Return ONLY a valid JSON array of question strings.
No preamble, no explanation, no markdown code fences.

CRITICAL RULES for every question:
1. The person is always a real individual asking for help with a personal struggle.
   They are coming to someone for advice — a friend, a counselor, or a pastor.
   Write from that perspective: vulnerable, personal, and grounded in real life.
2. NEVER mention specific instrument names, survey names, scale names, or academic
   frameworks by name. A real person seeking help would not say "I took the VIA survey"
   or "my PHQ-9 score is high" — they would just describe how they feel and what they
   are struggling with. Use the instruments only as inspiration for the scenario content,
   never as something the person references directly.
3. Every question must read as completely natural and conversational — like something
   a real person would actually say, not something from a research questionnaire.
4. You MUST generate exactly the number of questions specified. Count carefully.
   Do not stop early. If the instruction says 40, produce exactly 40.

Format:
["Question one here...", "Question two here...", ...]"""


# ── File loading ──────────────────────────────────────────────────────────────

def load_txt(path: Path) -> str:
    """Load a grounding txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_text(path: Path, max_chars: int = 15000) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) >= max_chars:
            break
    return text[:max_chars].strip()


def load_grounding(config: dict) -> str:
    """Load the grounding content for a dimension."""
    path = GROUNDING_BASE / config["grounding_subdir"] / config["grounding_file"]

    if not path.exists():
        raise FileNotFoundError(
            f"Grounding file not found: {path}\n"
            f"Make sure the file exists at that path."
        )

    if config["use_pdf"]:
        print(f"  Loading PDF: {path.name}")
        text = load_pdf_text(path)
    else:
        print(f"  Loading TXT: {path.name}")
        text = load_txt(path)

    print(f"  Loaded {len(text):,} characters.")
    return text


# ── GPT-4o generation ─────────────────────────────────────────────────────────

def build_user_prompt(dimension: str, grounding_text: str, n_questions: int) -> str:
    """Build the user prompt for GPT-4o."""
    return f"""Below is the grounding document for the "{dimension}" dimension.
Follow its instructions exactly to generate {n_questions} subjective questions.

{grounding_text}

Return ONLY a JSON array of {n_questions} question strings.
["...", "...", ...]"""


def call_gpt4o(messages: list) -> list[str]:
    """Single GPT-4o API call, returns parsed list of question strings."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.8,
        max_tokens=6000,
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


def generate_questions(dimension: str, config: dict) -> list[str]:
    """Call GPT-4o to generate questions, retrying automatically if count is short."""
    print(f"\n{'='*60}")
    print(f"  Dimension : {dimension}")
    print(f"  Target    : {config['n_questions']} questions")
    print(f"{'='*60}")

    grounding_text = load_grounding(config)
    n_target = config["n_questions"]

    # Initial call
    print(f"  Calling GPT-4o...", end=" ", flush=True)
    try:
        user_prompt = build_user_prompt(dimension, grounding_text, n_target)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        questions = call_gpt4o(messages)
        print(f"done. Got {len(questions)} questions.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n  [ERROR] JSON parse failed: {e}")
        return []

    # Retry loop — ask for missing questions if count is short
    max_retries = 3
    attempt = 0
    while len(questions) < n_target and attempt < max_retries:
        missing = n_target - len(questions)
        attempt += 1
        print(f"  Only {len(questions)}/{n_target} generated. Requesting {missing} more (attempt {attempt})...", end=" ", flush=True)

        retry_prompt = (
            f"You previously generated {len(questions)} questions for the '{dimension}' dimension "
            f"but {missing} more are needed to reach the target of {n_target}.\n\n"
            f"Here are the questions already generated (do NOT repeat these):\n"
            + "\n".join(f"- {q}" for q in questions)
            + f"\n\nGenerate exactly {missing} NEW questions that are different from the ones above. "
            f"Follow the same rules: first person, asking for help, no instrument names mentioned.\n\n"
            f"Return ONLY a JSON array of exactly {missing} question strings.\n"
            f'["...", "..."]'
        )

        try:
            retry_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ]
            new_questions = call_gpt4o(retry_messages)
            questions.extend(new_questions)
            print(f"done. Total now: {len(questions)}.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n  [ERROR] Retry parse failed: {e}")
            break

    # Trim if somehow over target
    if len(questions) > n_target:
        print(f"  Trimming from {len(questions)} to {n_target}.")
        questions = questions[:n_target]

    print(f"  Final count: {len(questions)}/{n_target}.")
    return questions


# ── CSV output ────────────────────────────────────────────────────────────────

def save_questions(dimension: str, questions: list[str]):
    """Save generated questions to CSV."""
    output_path = OUTPUT_DIR / f"{dimension}_subjective.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dimension", "question"])
        writer.writeheader()
        for i, q in enumerate(questions, start=1):
            writer.writerow({
                "id": f"{dimension}_{i:03d}",
                "dimension": dimension,
                "question": str(q).strip(),
            })

    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate FAI subjective questions using GPT-4o and grounding documents."
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        help=(
            "Which dimension to generate. Options: "
            + ", ".join(DIMENSION_CONFIG.keys())
            + ", or 'all'"
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between API calls when running all dimensions (default: 3.0)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    if args.dimension == "all":
        dimensions_to_run = list(DIMENSION_CONFIG.keys())
    elif args.dimension in DIMENSION_CONFIG:
        dimensions_to_run = [args.dimension]
    else:
        raise ValueError(
            f"Unknown dimension '{args.dimension}'.\n"
            f"Valid options: {list(DIMENSION_CONFIG.keys())} or 'all'"
        )

    print(f"\nFAI Subjective Question Generator")
    print(f"Dimensions to run : {dimensions_to_run}")
    print(f"Output directory  : {OUTPUT_DIR}\n")

    summary = []

    for i, dimension in enumerate(dimensions_to_run):
        config = DIMENSION_CONFIG[dimension]

        try:
            questions = generate_questions(dimension, config)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            summary.append((dimension, 0, "SKIPPED — file not found"))
            continue

        if questions:
            save_questions(dimension, questions)
            status = "OK" if len(questions) == config["n_questions"] else f"WARN: got {len(questions)}, expected {config['n_questions']}"
            summary.append((dimension, len(questions), status))
        else:
            summary.append((dimension, 0, "FAILED — parse error"))

        # Delay between calls when running all
        if args.dimension == "all" and i < len(dimensions_to_run) - 1:
            print(f"  Waiting {args.delay}s...")
            time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dim, count, status in summary:
        print(f"  {dim:<25} {count:>3} questions   [{status}]")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
