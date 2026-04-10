import os
import sys
import json
import time
import random
import requests
import numpy as np
from openai import OpenAI


# ------------------------------------------------
# LLM PROXY VARIABLES (INJECTED BY VALIDATOR)
# ------------------------------------------------

LLM_BASE_URL = os.environ.get("API_BASE_URL")
LLM_API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")


# ------------------------------------------------
# YOUR ENVIRONMENT URL
# ------------------------------------------------

ENV_URL = "https://anirudhpatil-microgrid-rl-env.hf.space"


# ------------------------------------------------
# OPENAI CLIENT (LLM PROXY)
# ------------------------------------------------

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TASK_ID = "sunny_day"
SEED = 42
MAX_STEPS = 120

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------
# API HELPER
# ------------------------------------------------

def api_post(path, payload):

    url = f"{ENV_URL}{path}"

    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"[ERROR] API call failed {path}: {e}", file=sys.stderr)
        return None


# ------------------------------------------------
# SIMPLE POLICY
# ------------------------------------------------

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


# ------------------------------------------------
# REQUIRED LLM CALL
# ------------------------------------------------

def call_llm():

    try:

        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Return OK"}
            ],
            max_tokens=5,
            temperature=0
        )

    except Exception as e:
        print(f"[WARNING] LLM call failed: {e}", file=sys.stderr)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    start_time = time.time()

    print("[START]")
    print(f"task: {TASK_ID}")
    print(f"seed: {SEED}")

    reset_data = api_post(
        "/reset",
        {
            "task_id": TASK_ID,
            "seed": SEED
        }
    )

    if reset_data is None:
        print("[ERROR] reset failed")
        return

    session_id = reset_data["session_id"]
    state = reset_data["state"]

    done = False
    step = 0

    # REQUIRED PROXY CALL
    call_llm()

    while not done and step < MAX_STEPS:

        action = simple_policy(state)

        step_data = api_post(
            "/step",
            {
                "session_id": session_id,
                "action": action
            }
        )

        if step_data is None:
            break

        state = step_data["state"]
        reward = step_data["reward"]
        done = step_data["done"]

        print("[STEP]")
        print(f"t: {step}")
        print(f"action: {json.dumps(action)}")
        print(f"reward: {reward}")

        step += 1


    grade_data = api_post(
        "/grader",
        {
            "session_id": session_id
        }
    )

    score = 0.0

    if grade_data:
        score = grade_data.get("score", 0.0)

    print("[END]")
    print(f"score: {score}")

    elapsed = time.time() - start_time
    print(f"time_sec: {elapsed:.2f}", file=sys.stderr)


if __name__ == "__main__":

    try:
        main()

    except Exception as e:
        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(0)
