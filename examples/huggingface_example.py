
from typing import Any

from huggingface_hub import login
from transformers import OpenAiAgent
from transformers import HfAgent
from transformers import load_tool

openai_key = ""
huggingface_token = ""
login(huggingface_token)


def get_agent(llm_type: str) -> Any:
    """
    load the agent that matches the LLM type.

    :param llm_type: large language model type
    :return: loaded agent
    """

    if llm_type == "openai":
        agent = OpenAiAgent(model="text-davinci-003", api_key=openai_key)
    elif llm_type == "hf_starcoder":
        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    elif llm_type == "hf_starcoder_base":
        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
    elif llm_type == "hf_open_assistant":
        agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
    else:
        raise NotImplementedError

    return agent


def remote_execution_example() -> None:
    """
    Remote execution

    :return:  None
    """

    agent = get_agent(llm_type="openai")

    image = agent.run("Draw me a picture of rivers and lakes", remote=True)
    image.show()

    image = agent.chat("Draw me a picture of rivers and lakes", remote=True)
    image.show()


def code_generation_example() -> None:
    """
    Code generation

    :return:  None
    """

    agent = get_agent(llm_type="openai")

    image = agent.run("Draw me a picture of rivers and lakes", return_code=True)
    image.show()

    image_generator = load_tool("huggingface-tools/text-to-image")
    image = image_generator(prompt="rivers and lakes")
    image.show()


def chat_example() -> None:
    """
    Chat-based execution (chat)

    :return:  None
    """

    agent = get_agent(llm_type="openai")
    image = agent.chat("Generate a picture of rivers and lakes")
    image.show()

    image = agent.chat("Transform the picture so that there is a rock in there")
    image.show()


def example() -> None:
    """
    Base example

    :return:  None
    """

    agent = get_agent(llm_type="openai")

    picture = agent.run("Draw me a picture of rivers and lakes.")
    picture.show()

    picture = agent.run("Draw me a picture of the sea then transform the picture to add an island")
    picture.show()

    picture = agent.run("Generate a picture of rivers and lakes.")
    picture.show()

    updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
    updated_picture.show()


if __name__ == '__main__':
    example()
