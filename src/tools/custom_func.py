
from typing import Any

from langchain.tools import tool, Tool
from langchain.utilities import PythonREPL


@tool
def save_huggingface_model(save_path: str, model: Any) -> str:
    """useful for when you need to save huggingface model to a specific path"""

    model.save_pretrained(save_path)

    return f"{save_path} : Model saved successfully."


def save_huggingface_model_tool() -> Tool:
    """
    return save_hugging_face Tool

    :return: Tool
    """
    return Tool(
            name="save_huggingface_model",
            func=save_huggingface_model,
            description="useful for when you need to save huggingface model to a specific path"
    )


def python_tool() -> Tool:
    """
    return python Tool

    :return: Tool
    """

    python_repl = PythonREPL()

    return Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run
    )
