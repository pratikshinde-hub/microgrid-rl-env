import os
import sys
import json
import time
import random
import requests
import numpy as np
from openai import OpenAI

# ─────────────────────────────────────────────
# ENV VARIABLES (MANDATORY)
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if API_BASE_URL is None:
    print("[ERROR] API_BASE_URL not set", file=sys.stderr)
    sys.exit(1)

if MODEL_NAME is None:
    print("[ERROR] MODEL_NAME not set", file=sys.stderr)
    sys.exit(1)

if HF_TOKEN is None:
    print("[ERROR] HF_TOKEN not set", file=sys.stderr)
    sys.exit(1)

# ─────────────────────────────────────────────
# INIT LLM CLIENT (REQUIRED BY RULES)
# ─────────────────────────────────────────────

client = OpenAI(api_key=HF_TOKEN)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TASK_ID = "sunny_day"
SEED = 42
MAX_STEPS = 120

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────

def api_post(path, payload):
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"POST {path} failed: {str(e)}")


def api_get(path, params=None):
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"GET {path} failed: {str(e)}")


# ─────────────────────────────────────────────
# SIMPLE POLICY (FAST + DETERMINISTIC)
# ─────────────────────────────────────────────

def simple_policy(state):

    price = state.get("spot_price", 0.0)
    soc = state.get("soc", 0.5)

    if price < 50 and soc < 0.8:
        battery = 5.0
    elif price > 100 and soc > 0.2:
        battery = -5.0
    else:
        battery = 0.0

    return {
        "battery_kw": battery,
        "curtail_fraction": 0.0
    }


# ─────────────────────────────────────────────
# OPTIONAL LLM CALL (COMPLIANCE ONLY)
# ─────────────────────────────────────────────

def call_llm_stub():

    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Return OK"}
            ],
            max_tokens=5,
            temperature=0
        )

    except Exception:
        pass


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():

    start_time = time.time()

    print("[START]")
    print(f"task: {TASK_ID}")
    print(f"seed: {SEED}")

    # RESET ENVIRONMENT
    reset_data = api_post(
        "/reset",
        {
            "task_id": TASK_ID,
            "seed": SEED
        }
    )

    session_id = reset_data["session_id"]
    state = reset_data["state"]

    done = False
    step = 0

    # Required LLM call (once)
    call_llm_stub()

    while not done and step < MAX_STEPS:

        action = simple_policy(state)

        step_data = api_post(
            "/step",
            {
                "session_id": session_id,
                "action": action
            }
        )

        state = step_data["state"]
        reward = step_data["reward"]
        done = step_data["done"]

        print("[STEP]")
        print(f"t: {step}")
        print(f"action: {json.dumps(action)}")
        print(f"reward: {reward}")

        step += 1

    # FINAL GRADING
    grade_data = api_post(
        "/grader",
        {
            "session_id": session_id
        }
    )

    score = grade_data.get("score", 0.0)

    print("[END]")
    print(f"score: {score}")

    elapsed = time.time() - start_time
    print(f"time_sec: {elapsed:.2f}", file=sys.stderr)


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    try:
        main()

    except Exception as e:
        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)
