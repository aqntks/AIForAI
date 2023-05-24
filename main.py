
import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = ""


def main():
    llm = OpenAI(temperature=0.9)
    text = "What would be a good company name for a company that makes colorful socks?"
    print(llm(text))


if __name__ == '__main__':
    main()
    