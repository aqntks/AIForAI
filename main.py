
import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_huggingface_tool
from src.tools import (
    CustomSearchTool,
    CustomLoadHuggingFaceModelTool,
    save_huggingface_model_tool
)


def main():
    llm = OpenAI(temperature=0)
    hf_tool = load_huggingface_tool("lysandre/hf-model-downloads")

    tools = [
        CustomSearchTool(),
        hf_tool,
        CustomLoadHuggingFaceModelTool(),
        save_huggingface_model_tool()
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("Please save the model that classifies dogs and cats")


if __name__ == '__main__':
    main()
