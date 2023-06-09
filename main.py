
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_huggingface_tool
from src.tools import (
    CustomSearchTool,
    CustomLoadHuggingFaceDatasetTool,
    CustomLoadHuggingFaceModelTool,
    load_tokenizer_tool,
    load_data_collator_tool,
    load_training_arguments_tool,
    train_model_tool,
    python_tool,
    save_huggingface_model_tool,
)


def main():
    llm = OpenAI(temperature=0)
    hf_tool = load_huggingface_tool("lysandre/hf-model-downloads")

    tools = [
        hf_tool,
        CustomSearchTool(),
        CustomLoadHuggingFaceDatasetTool(),
        CustomLoadHuggingFaceModelTool(),
        load_tokenizer_tool(),
        load_data_collator_tool(),
        load_training_arguments_tool(),
        train_model_tool(),
        python_tool(),
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("Please train the model that text classification")


if __name__ == '__main__':
    main()
