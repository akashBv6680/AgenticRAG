import streamlit as st
import os
import tempfile
import uuid
import requests
import re
from datetime import datetime
from typing import List

from tavily import TavilyClient
from langchain_tavily import TavilySearch

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

COLLECTION_NAME = "agentic_rag_documents"

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history[st.session_state.current_chat_id] = {
        'messages': st.session_state.messages,
        'title': "New Chat",
        'date': datetime.now()
    }

@st.cache_resource
def initialize_dependencies():
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"Dependency initialization error: {e}")
        st.stop()

if 'db_client' not in st.session_state or 'model' not in st.session_state:
    st.session_state.db_client, st.session_state.model = initialize_dependencies()

def get_collection():
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

@tool
def retrieve_documents(query: str) -> str:
    try:
        collection = get_collection()
        model = st.session_state.model
        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=5)
        return "\n".join(results['documents'][0])
    except Exception as e:
        return f"Error in document retrieval: {e}"

@tool
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def duckduckgo_search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)

def create_agent():
    prompt_template = hub.pull("hwchase17/react-chat")

    tavily_api_key = st.secrets.get("TAVILY_API_KEY")
    if not tavily_api_key:
        st.error("TAVILY_API_KEY not found in secrets.")
        st.stop()

    # Create TavilySearch tool instance
    tavily_search_tool = TavilySearch(max_results=5)

    tools = [
        retrieve_documents,
        calculator,
        duckduckgo_search,
        tavily_search_tool
    ]

    together_api_key = st.secrets.get("TOGETHER_API_KEY")
    if together_api_key:
        from langchain_community.llms import Together
        llm = Together(
            together_api_key=together_api_key,
            model="mistralai/Mistral-7B-Instruct-v0.2"
        )
    else:
        from langchain.llms import HuggingFaceHub
        llm = HuggingFaceHub(repo_id="google/flan-t5-small")

    agent = create_react_agent(llm, tools, prompt_template)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def clear_chroma_data():
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents: List[str]):
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]

    collection.add(documents=documents, embeddings=embeddings, ids=document_ids)
    st.toast("Documents processed and stored successfully!", icon="✅")

def is_valid_github_raw_url(url: str) -> bool:
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_executor = create_agent()
                try:
                    response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
                    final_response = response.get('output', 'An error occurred.')
                except Exception as e:
                    final_response = f"An error occurred: {e}"
                st.markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- Main UI ---
st.title("Agentic RAG Chat Flow with Tavily")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("API Key Configuration")
    st.markdown(
        """
        Upload your API keys in Streamlit Cloud Secrets:
        
        - `TAVILY_API_KEY` for Tavily AI
        - `TOGETHER_API_KEY` for Together AI (optional)
        """
    )

    st.header("Agentic RAG Chat Flow")
    if st.button("New Chat"):
        st.session_state.messages = []
        clear_chroma_data()
        st.session_state.chat_history = {}
        st.session_state.current_chat_id = None
        st.experimental_rerun()

    st.subheader("Chat History")
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        sorted_chat_ids = sorted(
            st.session_state.chat_history.keys(),
            key=lambda x: st.session_state.chat_history[x]['date'],
            reverse=True
        )
        for chat_id in sorted_chat_ids:
            chat_title = st.session_state.chat_history[chat_id]['title']
            date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
            if st.button(f"**{chat_title}** - {date_str}", key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                st.experimental_rerun()

# Document upload/processing section
with st.container():
    st.subheader("Add Context Documents")
    uploaded_files = st.file_uploader("Upload text files (.txt)", type="txt", accept_multiple_files=True)
    github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")

    if uploaded_files:
        if st.button("Process Files"):
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    file_contents = uploaded_file.read().decode("utf-8")
                    documents = split_documents(file_contents)
                    process_and_store_documents(documents)
                st.success("All files processed and stored successfully! You can now ask questions about their content.")

    if github_url and is_valid_github_raw_url(github_url):
        if st.button("Process URL"):
            with st.spinner("Fetching and processing file from URL..."):
                try:
                    response = requests.get(github_url)
                    response.raise_for_status()
                    file_contents = response.text
                    documents = split_documents(file_contents)
                    process_and_store_documents(documents)
                    st.success("File from URL processed! You can now chat about its contents.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching URL: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

display_chat_messages()
handle_user_input()
