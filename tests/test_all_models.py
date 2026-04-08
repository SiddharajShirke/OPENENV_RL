"""
test_all_models.py — Manual NVIDIA API connectivity test.

NOT a pytest unit test. Run directly:
    python tests/test_all_models.py

Tests each model in the global pool with a minimal API call.
"""

import os
import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

from baseline_openai import GLOBAL_MODEL_POOL, FREE_POOL

MODELS_TO_TEST = GLOBAL_MODEL_POOL.copy()

key1 = os.getenv("NVIDIA_API_KEY")
key2 = os.getenv("NVIDIA_API_KEY_2")


def test_model(model_name, api_key, label):
    if not api_key:
        return "SKIP (No API Key)"

    from openai import OpenAI
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    print(f"Testing {label} model: {model_name}...", end="", flush=True)

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Return the word 'OK' and nothing else."}],
            max_tokens=5,
            temperature=0.0,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content.strip()
        print(f" SUCCESS ({elapsed:.2f}s) -> '{content}'")
        return "PASS"
    except Exception as e:
        print(f" FAILED: {str(e)[:100]}")
        return f"FAIL: {str(e)[:100]}"


if __name__ == "__main__":
    results = {}

    print("\n=== Testing Primary/Backup Models (Key 1) ===")
    for m in MODELS_TO_TEST:
        results[m] = test_model(m, key1, "Primary")
        time.sleep(1)

    print("\n=== Testing Free Pool Models (Key 2) ===")
    for m in FREE_POOL:
        results[m] = test_model(m, key2 or key1, "Free")
        time.sleep(1)

    print("\n\n" + "=" * 50)
    print(f"{'Model Name':<50} | {'Status'}")
    print("-" * 70)
    for m, status in results.items():
        print(f"{m:<50} | {status}")
    print("=" * 50)

    fails = [m for m, s in results.items() if s.startswith("FAIL")]
    summary = f"Tested {len(results)} models. "
    if fails:
        summary += f"Found {len(fails)} failures: {', '.join(fails)}"
    else:
        summary += "All tests passed!"
    print(f"\nSummary: {summary}")
