"""Probe individual NIM chat models to find which actually work on this account."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.environ["NVIDIA_API_KEY"]
client = OpenAI(
    api_key=key,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=30.0,
    max_retries=0,
)

CANDIDATES = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-instruct",
]

for m in CANDIDATES:
    sys.stdout.write(f"  test  {m:55s} ... ")
    sys.stdout.flush()
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=10,
        )
        out = r.choices[0].message.content
        sys.stdout.write(f"OK   -> {out!r}\n")
    except Exception as e:
        msg = str(e).split("\n", 1)[0][:120]
        sys.stdout.write(f"FAIL -> {msg}\n")
    sys.stdout.flush()

print()
print("== embeddings ==")
for m in ["nvidia/nv-embedqa-e5-v5", "nvidia/nv-embed-v1", "nvidia/llama-3.2-nv-embedqa-1b-v1"]:
    for input_type in ["passage", "query"]:
        sys.stdout.write(f"  test  {m}  input_type={input_type:8s} ... ")
        sys.stdout.flush()
        try:
            r = client.embeddings.create(
                model=m,
                input="test text",
                extra_body={"input_type": input_type},
            )
            dim = len(r.data[0].embedding)
            sys.stdout.write(f"OK   dim={dim}\n")
        except Exception as e:
            msg = str(e).split("\n", 1)[0][:120]
            sys.stdout.write(f"FAIL -> {msg}\n")
        sys.stdout.flush()

