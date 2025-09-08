import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
import re
from typing import List

# --- Set Page Config (Must be the very first Streamlit command) ---
st.set_page_config(layout="wide")

# This block MUST be at the very top to fix the sqlite3 version issue.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed. Please add 'pysqlite3-binary' to your requirements.txt.")
    st.stop()

# Now import other libraries
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain import hub
from langchain_community.llms import Together
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun


# --- Constants and Configuration ---
COLLECTION_NAME = "agentic_rag_documents"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


# --- Centralized Session State Initialization ---
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
    """
    Initializes and returns the ChromaDB client and SentenceTransformer model.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"An error occurred during dependency initialization: {e}.")
        st.stop()

if 'db_client' not in st.session_state or 'model' not in st.session_state:
    st.session_state.db_client, st.session_state.model = initialize_dependencies()


def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

@tool
def retrieve_documents(query: str) -> str:
    """Searches for and returns documents relevant to the query from the vector database.
    This tool should be used when the user asks a question about the uploaded documents."""
    try:
        collection = get_collection()
        model = st.session_state.model
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )
        return "\n".join(results['documents'][0])
    except Exception as e:
        return f"An error occurred during document retrieval: {e}"

@tool
def calculator(expression: str) -> str:
    """Calculates the result of a mathematical expression string.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "2 * 3 + 5").
        This tool is useful for simple calculations."""
    try:
        # Use eval() with caution as it can be a security risk in production environments
        return str(eval(expression))
    except Exception as e:
        return f"Error: Could not evaluate expression. {e}"

@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web for the given query using DuckDuckGo.
    
    Args:
        query: The search query. This tool is useful for current events or general knowledge
        questions not covered by the documents.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

def create_agent():
    """Creates and returns a LangChain agent executor."""
    prompt_template = hub.pull("hwchase17/react-chat")
    
    # You can customize the tools the agent has access to here.
    # The agent will decide which tool to use based on the user's query.
    tools = [
        retrieve_documents, # For RAG functionality on uploaded docs
        calculator,         # For mathematical queries
        duckduckgo_search   # For general web search
    ]
    
    together_llm = Together(
        together_api_key=TOGETHER_API_KEY,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    agent = create_react_agent(together_llm, tools, prompt_template)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100) -> List[str]:
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents: List[str]):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )
    st.toast("Documents processed and stored successfully!", icon="✅")

def is_valid_github_raw_url(url: str) -> bool:
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
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
st.title("Agentic RAG Chat Flow")
st.markdown("---")

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
                    st.error(f"An unexpected error occurred: {e}")

# Sidebar
with st.sidebar:
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

display_chat_messages()
handle_user_input()
