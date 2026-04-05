import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Path bootstrap to import from parent or same dir if needed
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the model registry names from baseline_openai
try:
    import baseline_openai as b
    MODELS_TO_TEST = [
        b.MISTRAL_LARGE_2,
        b.MISTRAL_MIXTRAL_22B,
        b.KIMI_K2_INSTRUCT,
        b.KIMI_K2_INSTRUCT_0905,
        b.QWEN_72B,
        b.QWQ_32B,
        b.LLAMA_70B,
        b.LLAMA_405B,
        b.NEMOTRON_70B,
        b.PHI4_MINI,
    ]
    FREE_POOL = b.FREE_POOL
except ImportError:
    print("Could not import baseline_openai.py. Using hardcoded list.")
    MODELS_TO_TEST = [
        "mistralai/mistral-large-2-instruct",
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "qwen/qwen3-next-80b-a3b-instruct",
        "qwen/qwq-32b",
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-405b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "microsoft/phi-4-mini-instruct",
    ]
    FREE_POOL = [
        "qwen/qwen2-7b-instruct",
        "meta/llama-3.1-8b-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
        "microsoft/phi-3-mini-128k-instruct",
    ]

load_dotenv()

key1 = os.getenv("NVIDIA_API_KEY")
key2 = os.getenv("NVIDIA_API_KEY_2")

def test_model(model_name, api_key, label):
    if not api_key:
        return "SKIP (No API Key)"
    
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    print(f"Testing {label} model: {model_name}...", end="", flush=True)
    
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Return the word 'OK' and nothing else."}],
            max_tokens=5,
            temperature=0.0
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content.strip()
        print(f" SUCCESS ({elapsed:.2f}s) -> '{content}'")
        return "PASS"
    except Exception as e:
        print(f" FAILED: {str(e)[:100]}")
        return f"FAIL: {str(e)[:100]}"

results = {}

print("\n=== Testing Primary/Backup Models (Key 1) ===")
for m in MODELS_TO_TEST:
    results[m] = test_model(m, key1, "Primary")
    time.sleep(1) # Small delay between tests

print("\n=== Testing Free Pool Models (Key 2) ===")
for m in FREE_POOL:
    results[m] = test_model(m, key2 or key1, "Free")
    time.sleep(1)

print("\n\n" + "="*50)
print(f"{'Model Name':<50} | {'Status'}")
print("-" * 70)
for m, status in results.items():
    print(f"{m:<50} | {status}")
print("="*50)

summary = f"Tested {len(results)} models. "
fails = [m for m, s in results.items() if s.startswith("FAIL")]
if fails:
    summary += f"Found {len(fails)} failures: {', '.join(fails)}"
else:
    summary += "All tests passed!"
print(f"\nSummary: {summary}")
