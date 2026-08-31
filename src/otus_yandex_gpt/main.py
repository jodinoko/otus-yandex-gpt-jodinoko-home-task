import json
from pathlib import Path

from .prompts import build_messages

CHUNKS_PATH = Path("results/legal_test_chunks.jsonl")

def main() -> None:
    if not CHUNKS_PATH.exists():
        print("Сначала запустите: uv run prepare-legal-data")
        return

    with CHUNKS_PATH.open(encoding="utf-8") as chunks_file:
        first_chunk = json.loads(chunks_file.readline())

    messages = build_messages(first_chunk["text"])
    print(json.dumps(messages, ensure_ascii=False, indent=2))
