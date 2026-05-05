"""
fai_answer_objective_questions.py

Sends all objective questions to each of the 4 tested models and records
their answers. Each model picks A, B, C, or D for each question.

Models tested:
    - Claude Sonnet 4.6 (Anthropic direct)
    - Claude Sonnet 4.5 via Gloo (OAuth2 authenticated)
    - Gemini 2.5 Flash (Google direct)
    - Gemini 2.5 Flash via Gloo (OAuth2 authenticated)

Place this script in your project root (same level as data/).

Input:
    data/questions/*_objective.csv

Output:
    data/answers/{model_name}/{dimension}_objective_answers.csv

.env file required:
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    GLOO_CLIENT_ID=...
    GLOO_CLIENT_SECRET=...

Requirements:
    pip install anthropic google-generativeai openai python-dotenv pandas pyjwt requests
"""

import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directories ───────────────────────────────────────────────────────────────

QUESTIONS_DIR = Path("data/questions")
ANSWERS_DIR   = Path("data/answers")

# ── Gloo OAuth2 token manager ─────────────────────────────────────────────────

class GlooTokenManager:
    """
    Handles Gloo OAuth2 authentication.
    Fetches a new Bearer token using client credentials,
    and automatically refreshes it when it expires (tokens last 1 hour).
    """

    TOKEN_URL = "https://platform.ai.gloo.com/oauth2/token"
    GLOO_BASE_URL = "https://platform.ai.gloo.com/ai/v1"

    def __init__(self):
        self.client_id     = os.getenv("GLOO_CLIENT_ID")
        self.client_secret = os.getenv("GLOO_CLIENT_SECRET")
        self._token        = None
        self._token_expiry = 0  # Unix timestamp

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "GLOO_CLIENT_ID and GLOO_CLIENT_SECRET must be set in your .env file."
            )

    def get_token(self) -> str:
        """Return a valid Bearer token, fetching a new one if expired."""
        if self._token is None or time.time() >= self._token_expiry - 60:
            self._fetch_token()
        return self._token

    def _fetch_token(self):
        """Exchange client credentials for a Bearer token."""
        import base64
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        print("  [Gloo] Fetching new access token...", end=" ", flush=True)

        response = requests.post(
            self.TOKEN_URL,
            headers={
                "Content-Type":  "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data={
                "grant_type": "client_credentials",
                "scope":      "api/access",
            },
        )
        response.raise_for_status()
        token_data = response.json()

        self._token = token_data["access_token"]

        # Decode expiry from JWT payload
        try:
            from jwt import decode as jwt_decode
            decoded = jwt_decode(self._token, options={"verify_signature": False})
            self._token_expiry = decoded.get("exp", time.time() + 3600)
        except Exception:
            # Fallback: assume 1 hour expiry
            self._token_expiry = time.time() + 3600

        print("done.")

    def call(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Make a completion call to a Gloo model."""
        from openai import OpenAI
        client = OpenAI(
            api_key=self.get_token(),
            base_url=self.GLOO_BASE_URL,
        )
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=10,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# Shared Gloo token manager instance (created lazily)
_gloo_manager = None

def get_gloo_manager() -> GlooTokenManager:
    global _gloo_manager
    if _gloo_manager is None:
        _gloo_manager = GlooTokenManager()
    return _gloo_manager


# ── Model config ──────────────────────────────────────────────────────────────

MODELS = {
    "claude_sonnet_4_5": {
        "provider":    "anthropic",
        "model_id":    "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "is_gloo":     False,
    },
    "claude_sonnet_4_5_gloo": {
        "provider":    "gloo",
        "model_id":    "gloo-anthropic-claude-sonnet-4.5",   # Exact Gloo model ID
        "api_key_env": None,                   # Uses OAuth2, not a simple key
        "is_gloo":     True,
    },
    "gemini_2_5_flash": {
        "provider":    "google",
        "model_id":    "gemini-2.5-flash-preview-04-17",
        "api_key_env": "GOOGLE_API_KEY",
        "is_gloo":     False,
    },
    "gemini_2_5_flash_gloo": {
        "provider":    "gloo",
        "model_id":    "gloo-google-gemini-2.5-flash",    # Exact Gloo model ID
        "api_key_env": None,                   # Uses OAuth2, not a simple key
        "is_gloo":     True,
    },
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are taking a multiple choice exam.
For each question, respond with ONLY the letter of the correct answer: A, B, C, or D.
Do not explain your answer. Do not write anything else. Just the letter."""

def build_question_prompt(row: pd.Series) -> str:
    return (
        f"{row['question']}\n\n"
        f"A) {row['choice_a']}\n"
        f"B) {row['choice_b']}\n"
        f"C) {row['choice_c']}\n"
        f"D) {row['choice_d']}\n\n"
        f"Answer (A, B, C, or D only):"
    )


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(raw: str) -> str:
    """Extract a single letter A-D from model response."""
    raw = raw.strip().upper()
    if raw in ["A", "B", "C", "D"]:
        return raw
    match = re.search(r"\b([ABCD])\b", raw)
    if match:
        return match.group(1)
    if raw and raw[0] in ["A", "B", "C", "D"]:
        return raw[0]
    return "UNKNOWN"


# ── API callers ───────────────────────────────────────────────────────────────

def call_anthropic(model_id: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def call_google(model_id: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def call_model(model_name: str, config: dict, prompt: str) -> str:
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        return call_anthropic(model_id, prompt)
    elif provider == "google":
        return call_google(model_id, prompt)
    elif provider == "gloo":
        return get_gloo_manager().call(model_id, SYSTEM_PROMPT, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def answer_dimension(model_name: str, config: dict, dimension: str, df: pd.DataFrame, delay: float) -> pd.DataFrame:
    """Send all questions for one dimension to one model and record answers."""
    results = []
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        prompt = build_question_prompt(row)

        try:
            raw_response = call_model(model_name, config, prompt)
            extracted    = extract_answer(raw_response)
        except Exception as e:
            print(f"\n      [ERROR] {row['id']}: {e}")
            raw_response = "ERROR"
            extracted    = "ERROR"

        correct    = str(row["answer_label"]).upper()
        is_correct = (extracted == correct)

        results.append({
            "id":             row["id"],
            "dimension":      row["dimension"],
            "model":          model_name,
            "question":       row["question"],
            "correct_answer": correct,
            "raw_response":   raw_response,
            "extracted":      extracted,
            "is_correct":     is_correct,
        })

        # Progress indicator
        print(f"\r    Progress: {i+1}/{n} | Last: {row['id']} → {extracted} ({'✓' if is_correct else '✗'})", end="", flush=True)
        time.sleep(delay)

    print()  # newline after progress
    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Send FAI objective questions to each model and record answers."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Which model to run. Options: " + ", ".join(MODELS.keys()) + ", or 'all'",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        help="Which dimension to run (e.g. 'faith'), or 'all'",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between questions (default: 0.5)",
    )
    args = parser.parse_args()

    # Select models
    if args.model == "all":
        models_to_run = MODELS
    elif args.model in MODELS:
        models_to_run = {args.model: MODELS[args.model]}
    else:
        raise ValueError(f"Unknown model '{args.model}'. Options: {list(MODELS.keys())} or 'all'")

    # Find question files
    question_files = sorted(QUESTIONS_DIR.glob("*_objective.csv"))
    if args.dimension != "all":
        question_files = [f for f in question_files if f.stem.startswith(args.dimension)]

    if not question_files:
        print(f"No objective question files found in {QUESTIONS_DIR}")
        return

    print(f"\nFAI Objective Answer Generator")
    print(f"{'=' * 60}")
    print(f"Models     : {list(models_to_run.keys())}")
    print(f"Dimensions : {[f.stem.replace('_objective', '') for f in question_files]}")
    print(f"{'=' * 60}\n")

    summary = []

    for model_name, config in models_to_run.items():

        # Check API keys for non-Gloo models
        if not config["is_gloo"] and not os.getenv(config["api_key_env"]):
            print(f"[SKIP] {model_name} — missing {config['api_key_env']} in .env")
            continue

        # Check Gloo credentials
        if config["is_gloo"]:
            if not os.getenv("GLOO_CLIENT_ID") or not os.getenv("GLOO_CLIENT_SECRET"):
                print(f"[SKIP] {model_name} — missing GLOO_CLIENT_ID or GLOO_CLIENT_SECRET in .env")
                continue

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        model_output_dir = ANSWERS_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for question_file in question_files:
            dimension = question_file.stem.replace("_objective", "")
            print(f"\n  [{dimension}] Loading {question_file.name}...")

            df_questions = pd.read_csv(question_file)
            print(f"  [{dimension}] {len(df_questions)} questions. Answering...")

            df_answers = answer_dimension(model_name, config, dimension, df_questions, args.delay)

            output_path = model_output_dir / f"{dimension}_objective_answers.csv"
            df_answers.to_csv(output_path, index=False, encoding="utf-8")

            n_correct = df_answers["is_correct"].sum()
            n_total   = len(df_answers)
            accuracy  = n_correct / n_total * 100

            print(f"  [{dimension}] Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)")
            print(f"  [{dimension}] Saved → {output_path}")
            summary.append((model_name, dimension, n_correct, n_total, accuracy))

            time.sleep(1.0)  # pause between dimensions

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<28} {'Dimension':<25} {'Correct':>7} {'Total':>6} {'Accuracy':>9}")
    print("-" * 78)
    for model_name, dimension, n_correct, n_total, accuracy in summary:
        print(f"{model_name:<28} {dimension:<25} {n_correct:>7} {n_total:>6} {accuracy:>8.1f}%")


if __name__ == "__main__":
    main()
