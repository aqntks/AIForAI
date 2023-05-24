
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate


def chapter1() -> None:
    """
    LLMs: Get predictions from a language model

    :return: None
    """

    llm = OpenAI(temperature=0.9)
    text = "What would be a good company name for a company that makes colorful socks?"
    print(llm(text))


def chapter2() -> None:
    """
    Prompt Templates: Manage prompts for LLMs

    :return: None
    """

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product="colorful socks"))


def chapter3() -> None:
    """
    Chains: Combine LLMs and prompts in multi-step workflows

    :return: None
    """

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    chain.run("colorful socks")


def chapter4() -> None:
    """
    Agents: Dynamically Call Chains Based on User Input

    :return: None
    """

    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    agent.run(
        "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")


if __name__ == '__main__':
    chapter4()
