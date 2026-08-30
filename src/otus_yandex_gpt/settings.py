"""Load application settings from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Yandex Cloud credentials required by the application."""

    yc_api_key: str
    yc_folder_id: str


def load_settings() -> Settings:
    """Return validated settings loaded from the local ``.env`` file."""
    load_dotenv()

    yc_api_key = os.getenv("YC_API_KEY")
    yc_folder_id = os.getenv("YC_FOLDER_ID")

    if not yc_api_key or not yc_folder_id:
        missing_variables = [
            name
            for name, value in {
                "YC_API_KEY": yc_api_key,
                "YC_FOLDER_ID": yc_folder_id,
            }.items()
            if not value
        ]
        names = ", ".join(missing_variables)
        raise ValueError(f"Missing required environment variables: {names}")

    return Settings(yc_api_key=yc_api_key, yc_folder_id=yc_folder_id)
