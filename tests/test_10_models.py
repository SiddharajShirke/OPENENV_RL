import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

key1 = os.getenv("NVIDIA_API_KEY")

MODELS_10 = [
    "meta/llama-3.3-70b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "meta/llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-v3.2",
    "qwen/qwq-32b",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct"
]

def test_model(model_name, api_key):
    if not api_key:
        return "SKIP (No API Key)"
    
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    print(f"Testing {model_name:<40}... ", end="", flush=True)
    
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with 'OK' only."}],
            max_tokens=10,
            temperature=0.0,
            timeout=10
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content.strip()
        print(f"SUCCESS ({elapsed:.2f}s) -> '{content}'")
        return "PASS"
    except Exception as e:
        print(f"FAILED: {str(e)[:100]}")
        return f"FAIL: {str(e)[:100]}"

print("=== Testing 10-Model Sequence ===\n")
results = {}
for m in MODELS_10:
    results[m] = test_model(m, key1)
    time.sleep(1) # Small delay to avoid aggressive rate limits

print("\n" + "="*60)
fails = [m for m, s in results.items() if s.startswith("FAIL")]
if fails:
    print(f"Summary: Found {len(fails)} failures: {', '.join(fails)}")
else:
    print("Summary: All 10 models passed successfully!")
print("="*60)
