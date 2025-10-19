from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
db_path = "db"

prompt_template = """
Answer the user's question:
{question}

based on this information:

{knowledge}

If you don't find the answer to the user's question in this information, please reply: I can't tell you that."""

def to_ask():
    question = input("Write your question: ")

    embedding_function = OpenAIEmbeddings()

    db = Chroma(persist_directory=db_path,
                embedding_function=embedding_function
                )

    results = db.similarity_search_with_relevance_scores(question, k=4)
    if len(results) == 0 or results[0][1] < 0.7:
        print("I couldn't find any relevant information in the database.")
        return
    
    results_text = []
    for result in results:
        text = result[0].page_content
        results_text.append(text)


    knowledge = "\n\n----\n\n".join(results_text)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"question": question, "knowledge": knowledge})
    
    modelo = ChatOpenAI(model="gpt-5-nano")
    answer_text = modelo.invoke(prompt).content
    print("AI response: ", answer_text)



to_ask()

