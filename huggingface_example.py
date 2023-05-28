
from typing import Any

from huggingface_hub import login
from transformers import OpenAiAgent
from transformers import HfAgent
from transformers import load_tool

openai_key = ""
huggingface_token = ""
login(huggingface_token)


def get_agent(llm_type: str) -> Any:

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


def remote_execution_example():
    agent = get_agent(llm_type="openai")

    agent.run("Draw me a picture of rivers and lakes", remote=True)

    agent.chat("Draw me a picture of rivers and lakes", remote=True)


def code_generation_example():
    agent = get_agent(llm_type="openai")

    agent.run("Draw me a picture of rivers and lakes", return_code=True)

    image_generator = load_tool("huggingface-tools/text-to-image")

    image = image_generator(prompt="rivers and lakes")


def chat_example():

    agent = get_agent(llm_type="openai")
    agent.chat("Generate a picture of rivers and lakes")
    agent.chat("Transform the picture so that there is a rock in there")


def example():

    agent = get_agent(llm_type="openai")

    agent.run("Draw me a picture of rivers and lakes.")

    agent.run("Draw me a picture of the sea then transform the picture to add an island")

    picture = agent.run("Generate a picture of rivers and lakes.")
    updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)


if __name__ == '__main__':
    example()
