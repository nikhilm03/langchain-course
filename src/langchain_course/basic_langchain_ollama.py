from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")  
    summary_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text: {text}",
    )
    # llm = ChatOpenAI(model_name="gpt-5", temperature=0)  
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    text_to_summarize = "LangChain is a framework for developing applications powered by language models. It enables developers to build applications that can understand and generate human-like text."
    response = chain.invoke(input={"text": text_to_summarize})
    print("Summary:", response.content)


if __name__ == "__main__":
    main()
