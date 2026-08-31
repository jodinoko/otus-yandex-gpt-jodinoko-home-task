"""Load and split https://huggingface.co/datasets/TryDotAtwo/russian-legal-ner."""

import json
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset

DATASET_NAME = "TryDotAtwo/russian-legal-ner"
DATASET_REVISION = "d23f5380ccbd0b14c27efb8c9b7280d13cb4b85a"
TEST_DATA_URL = (
    f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/"
    f"{DATASET_REVISION}/data/test.parquet"
)

MAX_CHARS = 12_000  # Длиннее 12 000 — 736 документов (7,36%).
OVERLAP_CHARS = 500
OUTPUT_PATH = Path("results/legal_test_chunks.jsonl")


def load_test_dataset():
    return load_dataset(
        "parquet",
        data_files={"test": TEST_DATA_URL},
        split="test",
    )


def print_length_statistics(texts: list[str]) -> None:
    lengths = [len(text) for text in texts]
    long_documents = sum(length > MAX_CHARS for length in lengths)
    long_percent = long_documents / len(lengths) * 100

    print(f"Документов: {len(lengths)}")
    print(f"Минимальная длина: {min(lengths)}")
    print(f"Средняя длина: {mean(lengths):.2f}")
    print(f"Медиана: {median(lengths)}")
    print(f"Максимальная длина: {max(lengths)}")
    print(
        f"Длиннее {MAX_CHARS}: {long_documents} "
        f"({long_percent:.2f}%)"
    )


def find_chunk_end(text: str, start: int) -> int:
    end = min(start + MAX_CHARS, len(text))
    if end == len(text):
        return end

    search_start = start + MAX_CHARS // 2

    paragraph_end = text.rfind("\n\n", search_start, end)
    if paragraph_end != -1:
        return paragraph_end + 2

    sentence_end = max(
        text.rfind(mark, search_start, end) for mark in ".!?…"
    )
    if sentence_end != -1:
        return sentence_end + 1

    word_end = text.rfind(" ", search_start, end)
    if word_end != -1:
        return word_end + 1

    return end


def chunk_text(text: str) -> list[dict]:
    chunks = []
    start = 0

    while start < len(text):
        end = find_chunk_end(text, start)
        chunks.append(
            {
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )

        if end == len(text):
            break

        start = end - OVERLAP_CHARS

    return chunks


def save_chunks(dataset) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for document_id, document in enumerate(dataset):
            for chunk_id, chunk in enumerate(chunk_text(document["text"])):
                chunk["document_id"] = document_id
                chunk["chunk_id"] = chunk_id
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunks_count += 1

    return chunks_count


def main() -> None:
    dataset = load_test_dataset()
    print_length_statistics(dataset["text"])

    chunks_count = save_chunks(dataset)
    print(f"Фрагментов сохранено: {chunks_count}")
    print(f"Файл: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
