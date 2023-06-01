
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

    output = chain.run("colorful socks")
    print(output)


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


def chapter5() -> None:
    """
    Memory: Add State to Chains and Agents

    :return: None
    """

    from langchain import OpenAI, ConversationChain

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, verbose=True)

    output = conversation.predict(input="Hi there!")
    print(output)
    output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(output)


def chapter6() -> None:
    """
    Get Message Completions from a Chat Model

    :return: None
    """

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    chat = ChatOpenAI(temperature=0)

    chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ]
    output = chat(messages)
    print(output)
    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love programming.")
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love artificial intelligence.")
        ],
    ]
    result = chat.generate(batch_messages)
    print(result)

    # You can recover things like token usage from this LLMResult
    print(result.llm_output['token_usage'])


def chapter7() -> None:
    """
    Chat Prompt Templates

    :return: None
    """

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    chat = ChatOpenAI(temperature=0)

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    output = chat(chat_prompt.format_prompt(input_language="English", output_language="French",
                                   text="I love programming.").to_messages())

    print(output)


def chapter8() -> None:
    """
    Chat Prompt Templates

    :return: None
    """

    from langchain.chat_models import ChatOpenAI
    from langchain import LLMChain
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    chat = ChatOpenAI(temperature=0)

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    output = chain.run(input_language="English", output_language="French", text="I love programming.")
    print(output)


def chapter9() -> None:
    """
    Agents with Chat Models

    :return: None
    """

    from langchain.agents import load_tools
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

    # First, let's load the language model we're going to use to control the agent.
    chat = ChatOpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    output = agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
    print(output)


def chapter10() -> None:
    """
    Memory: Add State to Chains and Agents

    :return: None
    """

    from langchain.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )
    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    output = conversation.predict(input="Hi there!")
    print(output)
    output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(output)
    output = conversation.predict(input="Tell me about yourself.")
    print(output)


if __name__ == '__main__':
    chapter10()
