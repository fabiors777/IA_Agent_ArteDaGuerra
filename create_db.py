from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
folder_database = "base"


def create_db():
    docs = load_docs()
    chunks = chunks_divide(docs)
    vectorize_chunks(chunks)
   
def load_docs():
    loader = PyPDFDirectoryLoader(folder_database, glob="*.pdf" )
    docs = loader.load()
    return docs

def chunks_divide(docs):
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True
    )
    chunks = doc_splitter.split_documents(docs)
    print(len(chunks))
    return chunks

def vectorize_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("Database created!")


create_db()