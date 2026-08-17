"""Discover which NVIDIA NIM models are accessible to this account."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

try:
    import requests
except ImportError:
    print("requests is required: pip install requests", file=sys.stderr)
    sys.exit(1)

key = os.environ.get("NVIDIA_API_KEY")
if not key:
    print("NVIDIA_API_KEY not set", file=sys.stderr)
    sys.exit(1)

r = requests.get(
    "https://integrate.api.nvidia.com/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
print(f"HTTP {r.status_code}")
try:
    d = r.json()
except Exception as e:
    print(f"Failed to parse JSON: {e}")
    print(r.text[:500])
    sys.exit(1)

ids = sorted(m.get("id", "") for m in d.get("data", []))
print(f"\n{len(ids)} models accessible:\n")

# Group by topic for readability
def topic(i: str) -> str:
    i = i.lower()
    if "nemotron" in i:
        return "nemotron"
    if "llama" in i and "nemotron" not in i:
        return "meta llama"
    if "mistral" in i:
        return "mistral"
    if "qwen" in i:
        return "qwen"
    if "gemma" in i:
        return "gemma"
    if "phi" in i:
        return "phi"
    if "nv-embed" in i or "nvembed" in i or "arctic" in i:
        return "embeddings"
    return "other"

groups: dict[str, list[str]] = {}
for mid in ids:
    groups.setdefault(topic(mid), []).append(mid)

for g in ["nemotron", "meta llama", "mistral", "qwen", "embeddings", "gemma", "phi", "other"]:
    if g in groups:
        print(f"== {g} ==")
        for m in groups[g]:
            print(f"  {m}")
        print()
