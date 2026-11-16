import time
from typing import List, Dict
from pathlib import Path
import streamlit as st
from humanfriendly import format_timespan
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Configurations
CHROMA_DIR = Path("unity_index")
COLLECTION_NAME = "unity_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLAMA_MODEL = "llama3.2"
TOP_K = 3

@st.cache_resource(show_spinner="Loading vector index & embedding model...")
def load_resources():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)
    return collection, embedder

try:
    collection, embedder = load_resources()
except Exception as e:
    st.error(f"Could not load resources. Is 'unity_index' populated? Error: {e}")

def retrieve_docs(query: str, k: int = TOP_K) -> List[Dict]:
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        docs.append({"content": doc, "meta": meta, "distance": dist})
    return docs

def build_prompt(query: str, docs: List[Dict], history: List[Dict]):
    system_msg = {
        "role": "system",
        "content": (
            "You are GameDev Assistant, an expert on Unity game engine.\n"
            "Answer strictly based on the provided documentation excerpts.\n"
            "If unsure, say you don't know. Include citations as [#] linking to the URLs."
        ),
    }
    context_lines = []
    url_map = []
    for idx, doc in enumerate(docs, 1):
        title = doc["meta"].get("title", "Docs")
        url = doc["meta"].get("url", "")
        snippet = doc["content"]
        context_lines.append(f"[#${idx}] {title}\n{snippet}")
        url_map.append(url)

    context_block = "\n\n".join(context_lines)
    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUERY: {query}\n\n"
        f"Respond with Markdown. Cite sources as [#]."
    )
    messages = [system_msg] + history + [{"role": "user", "content": user_content}]
    return messages, url_map

def stream_response(messages: List[Dict]):
    stream = ollama.chat(model=LLAMA_MODEL, messages=messages, stream=True)
    for part in stream:
        yield part["message"]["content"]

st.set_page_config(page_title="GameDev Assistant", layout="wide")
st.title("GameDev Assistant - Unity Docs RAG")

if "history" not in st.session_state:
    st.session_state.history = []

# Display previous conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about Unity...")
if query:
    st.chat_message("user").markdown(query)
    start_time = time.perf_counter()
    
    with st.spinner("Retrieving documents..."):
        docs = retrieve_docs(query, TOP_K)
        messages, url_map = build_prompt(query, docs, st.session_state.history)

    placeholder = st.chat_message("assistant")
    answer_container = placeholder.empty()
    partial = ""
    for chunk in stream_response(messages):
        partial += chunk
        answer_container.markdown(partial + "|")
    
    exec_ms = (time.perf_counter() - start_time) * 1000
    unique_urls = list(dict.fromkeys(url_map))
    refs_md = "\n\n**References:**\n" + "\n".join(f"- [#${i+1}]: {url}" for i, url in enumerate(unique_urls))
    final_answer = partial + refs_md + f"\n\n_Answered in {format_timespan(exec_ms / 1000)}_"
    answer_container.markdown(final_answer)

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": final_answer})

    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]

# --- Tunnel script to serve Streamlit in Colab ---
# Run this in bash next: 
# !npm install -g localtunnel
# !streamlit run app.py & npx localtunnel --port 8501
