import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    print("Error: NVIDIA_API_KEY not found in .env file.")
    # Fallback to check if NVIDIA_API_KEY_2 exists
    api_key = os.getenv("NVIDIA_API_KEY_2")
    if not api_key:
        print("Error: Neither NVIDIA_API_KEY nor NVIDIA_API_KEY_2 found in .env file.")
        exit(1)
    else:
        print("Using NVIDIA_API_KEY_2...")

print(f"Using API Key: {api_key[:10]}...{api_key[-5:]}")

# Initialize the OpenAI client with NVIDIA base URL
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

print("\nFetching models from NVIDIA API...")
try:
    # List models
    models = client.models.list()

    # Sort models by ID for better readability
    sorted_models = sorted(models.data, key=lambda x: x.id)

    output_lines = []
    output_lines.append(f"{'#':<4} | {'Model ID':<60}")
    output_lines.append("-" * 70)
    for i, m in enumerate(sorted_models, 1):
        output_lines.append(f"{i:<4} | {m.id:<60}")
    
    output_lines.append(f"\nSuccessfully listed {len(models.data)} models.")
    
    # Save to file with UTF-8 encoding
    with open("nvidia_models.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    # Also print it
    print("\n".join(output_lines))

except Exception as e:
    print(f"Error accessing NVIDIA API: {e}")
