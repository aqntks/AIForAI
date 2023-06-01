
from typing import Any

from langchain.tools import tool


@tool
def save_huggingface_model(save_path: str, model: Any) -> str:
    """useful for when you need to save huggingface model to a specific path"""

    model.save_pretrained(save_path)

    return f"{save_path} : Model saved successfully."
