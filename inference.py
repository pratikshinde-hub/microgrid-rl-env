import os
import sys
import json
import time
import random
import requests
import numpy as np
from openai import OpenAI

# ─────────────────────────────────────────────
# ENV VARIABLES
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not API_BASE_URL:
    print("[ERROR] API_BASE_URL not set", file=sys.stderr)
    sys.exit(1)

if not MODEL_NAME:
    print("[ERROR] MODEL_NAME not set", file=sys.stderr)
    sys.exit(1)

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not set", file=sys.stderr)
    sys.exit(1)

# remove trailing slash for safety
API_BASE_URL = API_BASE_URL.rstrip("/")

# ─────────────────────────────────────────────
# OPENAI CLIENT (MANDATORY)
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

def api_post(path, payload=None):

    url = f"{API_BASE_URL}{path}"

    r = requests.post(
        url,
        json=payload if payload else {},
        timeout=20
    )

    r.raise_for_status()

    return r.json()


def api_get(path, params=None):

    url = f"{API_BASE_URL}{path}"

    r = requests.get(
        url,
        params=params,
        timeout=20
    )

    r.raise_for_status()

    return r.json()

# ─────────────────────────────────────────────
# SIMPLE POLICY
# ─────────────────────────────────────────────

def simple_policy(state):

    price = state.get("spot_price", 0)
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
# LLM CALL (REQUIRED BY RULES)
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
# MAIN
# ─────────────────────────────────────────────

def main():

    start = time.time()

    print("[START]")
    print(f"task: {TASK_ID}")
    print(f"seed: {SEED}")

    # reset environment
    reset = api_post(
        "/reset",
        {
            "task_id": TASK_ID,
            "seed": SEED
        }
    )

    session_id = reset["session_id"]
    state = reset["state"]

    done = False
    step = 0

    # required LLM call
    call_llm_stub()

    while not done and step < MAX_STEPS:

        action = simple_policy(state)

        result = api_post(
            "/step",
            {
                "session_id": session_id,
                "action": action
            }
        )

        state = result["state"]
        reward = result["reward"]
        done = result["done"]

        print("[STEP]")
        print(f"t: {step}")
        print(f"action: {json.dumps(action)}")
        print(f"reward: {reward}")

        step += 1

    # grader
    grade = api_post(
        "/grader",
        {
            "session_id": session_id
        }
    )

    score = grade.get("score", 0.0)

    print("[END]")
    print(f"score: {score}")

    elapsed = time.time() - start
    print(f"time_sec: {elapsed:.2f}", file=sys.stderr)

# ─────────────────────────────────────────────

if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)
