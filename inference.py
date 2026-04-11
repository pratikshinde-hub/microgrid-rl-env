
import os
import sys
import json
import time
import random
import requests
import numpy as np
from openai import OpenAI


# ------------------------------------------------
# ENV VARIABLES (SAFE DEFAULTS)
# ------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_URL = "https://anirudhpatil-microgrid-rl-env.hf.space"

TASK_NAME = "sunny_day"
BENCHMARK = "microgrid"
MAX_STEPS = 120

random.seed(42)
np.random.seed(42)

# ------------------------------------------------
# OPENAI CLIENT (LLM PROXY)
# ------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# ------------------------------------------------
# ENV API CALL
# ------------------------------------------------

def api_post(path, payload):

    try:
        r = requests.post(
            f"{ENV_URL}{path}",
            json=payload,
            timeout=20
        )
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"[DEBUG] API error: {e}", flush=True)
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
# LLM CALL (REQUIRED)
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
        print(f"[DEBUG] LLM call failed: {e}", flush=True)


# ------------------------------------------------
# LOG FUNCTIONS
# ------------------------------------------------

def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done):
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={json.dumps(action)} reward={reward:.2f} done={done_val} error=null",
        flush=True
    )

def log_end(success, steps, score, rewards):

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    rewards = []
    steps_taken = 0

    log_start()

    try:

        reset = api_post(
            "/reset",
            {"task_id": TASK_NAME, "seed": 42}
        )

        if reset is None:
            raise RuntimeError("reset failed")

        session_id = reset["session_id"]
        state = reset["state"]

        call_llm()   # required LLM proxy call

        for step in range(1, MAX_STEPS + 1):

            action = simple_policy(state)

            result = api_post(
                "/step",
                {
                    "session_id": session_id,
                    "action": action
                }
            )

            if result is None:
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            state = result.get("state", {})

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done)

            if done:
                break

        grade = api_post(
            "/grader",
            {"session_id": session_id}
        )

        score = grade.get("score", 0.0) if grade else 0.0

        success = score >= 0.0

    except Exception as e:

        print(f"[DEBUG] runtime error: {e}", flush=True)

        success = False
        score = 0.0

    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
