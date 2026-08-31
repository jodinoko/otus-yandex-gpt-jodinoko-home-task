import json
from pathlib import Path

from .settings import load_settings
from .yandex_gpt import extract_entities

CHUNKS_PATH = Path("results/legal_test_chunks.jsonl")
REPORT_PATH = Path("results/evaluation_report.jsonl")
DATASET_DOCUMENT_IDS = {0, 57, 68, 1327, 1882}

NO_DATA_DOCUMENT = """
ДОВЕРЕННОСТЬ

ООО «Ромашка» уполномочивает Иванова Ивана Ивановича представлять
интересы организации и подписывать заявления от её имени.
""".strip()

NOISY_OCR_DOCUMENT = """
Д0ГОВ0Р № 17

ООО «Альфа», ИНН 7701234567, КПП 770101001, именуемое «Заказчик», и
000 «Бета», ИНН 7812345678, КПП 781201001, именуемое «Исполнитель»,
заключили настоящий догов0р. Цена раб0т — 125 000 рублeй.
Ср0к вып0лнения — 30 днeй с даты п0дписания.
""".strip()


def load_evaluation_documents() -> list[dict]:
    documents = []

    with CHUNKS_PATH.open(encoding="utf-8") as chunks_file:
        for line in chunks_file:
            chunk = json.loads(line)
            document_id = chunk["document_id"]
            if document_id in DATASET_DOCUMENT_IDS and chunk["chunk_id"] == 0:
                documents.append(
                    {
                        "name": f"dataset_document_{document_id}",
                        "kind": "dataset",
                        "text": chunk["text"],
                    }
                )

    documents.sort(key=lambda document: int(document["name"].rsplit("_", 1)[1]))
    documents.extend(
        [
            {
                "name": "no_amount_or_term",
                "kind": "no_data",
                "text": NO_DATA_DOCUMENT,
            },
            {
                "name": "noisy_ocr",
                "kind": "noisy_ocr",
                "text": NOISY_OCR_DOCUMENT,
            },
        ]
    )
    return documents


def main() -> None:
    if not CHUNKS_PATH.exists():
        print("Сначала запустите: uv run prepare-legal-data")
        return

    settings = load_settings()
    documents = load_evaluation_documents()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        for number, document in enumerate(documents, start=1):
            print(f"[{number}/{len(documents)}] {document['name']}")

            try:
                result = extract_entities(
                    document["text"],
                    api_key=settings.yc_api_key,
                    folder_id=settings.yc_folder_id,
                )
                report_row = {**document, "result": result}
            except (RuntimeError, ValueError) as error:
                report_row = {**document, "error": str(error)}

            report_file.write(json.dumps(report_row, ensure_ascii=False) + "\n")

    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
