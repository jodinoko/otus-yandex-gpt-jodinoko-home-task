import json
import re

import requests

from .prompts import build_messages

API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
TEMPERATURE = 0.2
MAX_TOKENS = 2_000


def build_request_body(document_text: str, folder_id: str) -> dict:
    return {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": TEMPERATURE,
            "maxTokens": str(MAX_TOKENS),
        },
        "messages": build_messages(document_text),
        "jsonObject": True,
    }


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # На случай, если модель всё же добавила текст до или после JSON.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

    return cleaned


def parse_json_response(text: str) -> dict:
    try:
        result = json.loads(clean_json_text(text))
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON") from error

    if not isinstance(result, dict):
        raise ValueError("Not JSON object was returned")

    return result


def extract_entities(document_text: str, api_key: str, folder_id: str) -> dict:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        },
        json=build_request_body(document_text, folder_id),
        timeout=120,
    )
    response.raise_for_status()

    try:
        answer_text = response.json()["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Invalid structure YandexGPT") from error

    return parse_json_response(answer_text)
