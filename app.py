# app.py
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- Config ----------
load_dotenv()
DB_PATH = "db"
RELEVANCE_THRESHOLD = 0.7
MODEL_NAME = "gpt-5-nano"  # igual ao seu código original

PROMPT_TMPL = """
Responda a pergunta do usuário:
{question}

com base nesta informação:

{knowledge}

Se você não encontrar a resposta para a pergunta do usuário nesta informação, responda: Eu não sei responder isso.
"""

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Sun Tzu Chat (RAG)", page_icon="📖")
st.title("📖 Sun Tzu • A Arte da Guerra")
st.caption("Respostas **apenas** com base no conteúdo do livro.")

# Inicia histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Pergunte algo sobre *A Arte da Guerra*."}
    ]

# ---------- Helpers (com cache p/ manter simples e rápido) ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner=False)
def get_db(_emb):
    return Chroma(persist_directory=DB_PATH, embedding_function=_emb)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model=MODEL_NAME)

def retrieve(question: str, k: int = 4):
    emb = get_embeddings()
    db = get_db(emb)
    return db.similarity_search_with_relevance_scores(question, k=k)

def build_prompt(question: str, chunks):
    texts = [doc.page_content for doc, _score in chunks]
    knowledge = "\n\n----\n\n".join(texts)
    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)
    return prompt.invoke({"question": question, "knowledge": knowledge})

def answer_question(question: str):
    results = retrieve(question, k=4)
    if len(results) == 0 or results[0][1] < RELEVANCE_THRESHOLD:
        return "Não sei responder isso.", results

    prompt = build_prompt(question, results)
    llm = get_llm()
    resp = llm.invoke(prompt).content
    return resp, results

# ---------- Render histórico ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Entrada do usuário ----------
if user_q := st.chat_input("Digite sua pergunta..."):
    # mostra a pergunta
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # gera a resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando a base e gerando resposta..."):
            try:
                resp, results = answer_question(user_q)
            except Exception as e:
                resp = f"Erro ao processar sua pergunta: `{e}`"
                results = []

        st.markdown(resp)
        # fonte dos trechos (opcional, mas útil)
        if results:
            with st.expander("Ver trechos usados (fontes)"):
                for i, (doc, score) in enumerate(results, start=1):
                    st.markdown(
                        f"**Trecho {i}** · relevância: `{score:.2f}`\n\n"
                        f"> {doc.page_content[:600]}{'...' if len(doc.page_content) > 600 else ''}"
                    )

    # salva resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": resp})

# ---------- Rodapé ----------
st.markdown("---")
st.caption(
    "Este agente responde **somente** com base no livro."
    "Se não encontrar, retorna *Eu não sei responder isso.*"
)
