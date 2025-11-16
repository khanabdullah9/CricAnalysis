from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os
from dotenv import load_dotenv
from vector import VectorEmbeddings
from html_reader import HTMLReader
from etl_handler import ETLHandler

class RAGInterface():
    def __init__(self, search_kwargs = 20):
        self.llm = create_agent(
                model="gpt-4o",
                system_prompt="You are a helpful assistant",
            )

        self.prompt = ChatPromptTemplate.from_template(
                """
                You are an expert in answering questions from a text data about Indian international cricketers

                Here is relevant text: {text}

                Here is the question to answer: {question}
                """
            )
        self.chain = self.prompt | self.llm
        self.search_kwargs = search_kwargs

    def generate(self, user_input):
        handler = ETLHandler()
        collection = handler.get_or_create_chroma_collection()

        query_results = collection.query(
            query_texts=[user_input],
            n_results = self.search_kwargs
        )

        response = ""
        try:
            rag_response = self.chain.invoke({
            "text": query_results,
            "question": user_input
        })
            response = rag_response["messages"][1].content
        except Exception as e:
            print(f"[ERR]: {e}")

        return response
