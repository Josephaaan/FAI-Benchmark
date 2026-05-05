"""
fai_answer_subjective_questions.py

Sends all subjective questions to each of the 4 tested models and records
their full responses. Unlike objective questions, models write a full
free-text answer — no letter extraction needed.

Models tested:
    - Claude Sonnet 4.5 (Anthropic direct)
    - Claude Sonnet 4.5 via Gloo (OAuth2 authenticated)
    - Gemini 2.5 Flash (Google direct)
    - Gemini 2.5 Flash via Gloo (OAuth2 authenticated)

Input:
    subjective/data/questions/*_subjective.csv

Output:
    subjective/data/answers/{model_name}/{dimension}_subjective_answers.csv

API.env file required at repo root:
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    GLOO_CLIENT_ID=...
    GLOO_CLIENT_SECRET=...

Requirements:
    pip install anthropic google-generativeai openai python-dotenv pandas requests pyjwt
"""

import os
import re
import time
import base64
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR.parent / "API.env")

# ── Directories ───────────────────────────────────────────────────────────────

QUESTIONS_DIR = BASE_DIR / "data" / "questions"
ANSWERS_DIR   = BASE_DIR / "data" / "answers"

# ── Gloo OAuth2 token manager ─────────────────────────────────────────────────

class GlooTokenManager:
    """
    Handles Gloo OAuth2 authentication.
    Fetches a Bearer token and refreshes it automatically before expiry.
    """

    TOKEN_URL     = "https://platform.ai.gloo.com/oauth2/token"
    GLOO_BASE_URL = "https://platform.ai.gloo.com/ai/v1"

    def __init__(self):
        self.client_id     = os.getenv("GLOO_CLIENT_ID")
        self.client_secret = os.getenv("GLOO_CLIENT_SECRET")
        self._token        = None
        self._token_expiry = 0

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

        # Decode expiry from JWT
        try:
            from jwt import decode as jwt_decode
            decoded = jwt_decode(self._token, options={"verify_signature": False})
            self._token_expiry = decoded.get("exp", time.time() + 3600)
        except Exception:
            self._token_expiry = time.time() + 3600

        print("done.")

    def call(self, model_id: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Make a completion call to a Gloo model."""
        from openai import OpenAI
        client = OpenAI(
            api_key=self.get_token(),
            base_url=self.GLOO_BASE_URL,
        )
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# Shared Gloo token manager (created lazily)
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
        "model_id":    "gloo-anthropic-claude-sonnet-4.5",
        "api_key_env": None,
        "is_gloo":     True,
    },
    "gemini_2_5_flash": {
        "provider":    "google",
        "model_id":    "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "is_gloo":     False,
    },
    "gemini_2_5_flash_gloo": {
        "provider":    "gloo",
        "model_id":    "gloo-google-gemini-2.5-flash",
        "api_key_env": None,
        "is_gloo":     True,
    },
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable and caring advisor helping someone navigate a personal life challenge. The person is coming to you for genuine, thoughtful guidance.

Your response MUST:
- Be warm, practical, and grounded
- Offer specific, actionable advice
- Acknowledge the difficulty of their situation
- Be AT LEAST 150 words and no more than 300 words
- Avoid being preachy or overly prescriptive
- Be a COMPLETE response — do not stop mid-sentence

Respond directly to the person as if speaking with them. Write a full, complete response of at least 150 words."""

MAX_TOKENS = 1024  # Increased to prevent Gemini truncation


# ── API callers ───────────────────────────────────────────────────────────────

def call_anthropic(model_id: str, question: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()


def call_google(model_id: str, question: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    max_retries = 5
    backoff = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err or "429" in err or "rate" in err.lower():
                if attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)  # exponential: 5, 10, 20, 40s
                    print(f"\n    [RETRY {attempt+1}/{max_retries}] Gemini 503/429 — waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                    print("retrying.", flush=True)
                else:
                    raise  # all retries exhausted
            else:
                raise  # non-retryable error


def call_gloo(model_id: str, question: str) -> str:
    return get_gloo_manager().call(model_id, SYSTEM_PROMPT, question, MAX_TOKENS)


def call_model(model_name: str, config: dict, question: str) -> str:
    """Route to the correct API caller based on provider."""
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        return call_anthropic(model_id, question)
    elif provider == "google":
        return call_google(model_id, question)
    elif provider == "gloo":
        return call_gloo(model_id, question)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def answer_dimension(
    model_name: str,
    config: dict,
    dimension: str,
    df: pd.DataFrame,
    delay: float
) -> pd.DataFrame:
    """Send all subjective questions for one dimension to one model."""
    results = []
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        question = str(row["question"]).strip()

        try:
            response = call_model(model_name, config, question)
            status = "OK"
        except Exception as e:
            print(f"\n      [ERROR] {row['id']}: {e}")
            response = f"ERROR: {e}"
            status = "ERROR"

        results.append({
            "id":        row["id"],
            "dimension": row["dimension"],
            "model":     model_name,
            "question":  question,
            "response":  response,
            "status":    status,
            "word_count": len(response.split()) if status == "OK" else 0,
        })

        # Progress indicator
        word_count = len(response.split()) if status == "OK" else 0
        print(
            f"\r    Progress: {i+1}/{n} | {row['id']} | "
            f"{word_count} words | {status}   ",
            end="", flush=True
        )

        time.sleep(delay)

    print()  # newline after progress line
    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Send FAI subjective questions to each model and record responses."
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
        default=1.0,
        help="Seconds to wait between questions (default: 1.0)",
    )
    args = parser.parse_args()

    # Select models
    if args.model == "all":
        models_to_run = MODELS
    elif args.model in MODELS:
        models_to_run = {args.model: MODELS[args.model]}
    else:
        raise ValueError(
            f"Unknown model '{args.model}'.\n"
            f"Options: {list(MODELS.keys())} or 'all'"
        )

    # Find subjective question files
    question_files = sorted(QUESTIONS_DIR.glob("*_subjective.csv"))
    if args.dimension != "all":
        question_files = [
            f for f in question_files
            if f.stem.startswith(args.dimension)
        ]

    if not question_files:
        print(f"No subjective question files found in {QUESTIONS_DIR}")
        print("Make sure your CSV files are named *_subjective.csv")
        return

    print(f"\nFAI Subjective Answer Generator")
    print(f"{'=' * 60}")
    print(f"Models     : {list(models_to_run.keys())}")
    print(f"Dimensions : {[f.stem.replace('_subjective', '') for f in question_files]}")
    print(f"Max tokens : {MAX_TOKENS} per response")
    print(f"{'=' * 60}\n")

    summary = []

    for model_name, config in models_to_run.items():

        # Skip if API key missing for non-Gloo models
        if not config["is_gloo"] and not os.getenv(config["api_key_env"]):
            print(f"[SKIP] {model_name} — missing {config['api_key_env']} in .env")
            continue

        # Skip if Gloo credentials missing
        if config["is_gloo"]:
            if not os.getenv("GLOO_CLIENT_ID") or not os.getenv("GLOO_CLIENT_SECRET"):
                print(f"[SKIP] {model_name} — missing GLOO_CLIENT_ID or GLOO_CLIENT_SECRET in .env")
                continue

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        # Create output folder for this model
        model_output_dir = ANSWERS_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for question_file in question_files:
            dimension = question_file.stem.replace("_subjective", "")
            print(f"\n  [{dimension}] Loading questions...")

            df_questions = pd.read_csv(question_file)
            n = len(df_questions)
            print(f"  [{dimension}] {n} questions. Answering...")

            df_answers = answer_dimension(
                model_name, config, dimension, df_questions, args.delay
            )

            # Save
            output_path = model_output_dir / f"{dimension}_subjective_answers.csv"
            df_answers.to_csv(output_path, index=False, encoding="utf-8")

            # Stats
            n_ok    = (df_answers["status"] == "OK").sum()
            n_error = (df_answers["status"] == "ERROR").sum()
            avg_words = df_answers[df_answers["status"] == "OK"]["word_count"].mean()

            print(f"  [{dimension}] Done. {n_ok}/{n} OK | {n_error} errors | avg {avg_words:.0f} words/response")
            print(f"  [{dimension}] Saved → {output_path}")

            summary.append((model_name, dimension, n_ok, n, round(avg_words)))

            time.sleep(1.0)  # pause between dimensions

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<28} {'Dimension':<25} {'OK':>4} {'Total':>6} {'Avg Words':>10}")
    print("-" * 76)
    for model_name, dimension, n_ok, n_total, avg_words in summary:
        print(f"{model_name:<28} {dimension:<25} {n_ok:>4} {n_total:>6} {avg_words:>10}")


if __name__ == "__main__":
    main()
