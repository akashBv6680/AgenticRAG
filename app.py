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
import shutil
from pathlib import Path
from typing import List, Dict, Any

# This block MUST be at the very top to fix the sqlite3 version issue.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed. Please add 'pysqlite3-binary' to your requirements.txt.")
    st.stop()

# Now import chromadb and other libraries
import chromadb
from langchain_community.llms import Together
from langchain_together.embeddings import TogetherEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
TOGETHER_API_KEY = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Dictionary of supported languages and their ISO 639-1 codes for the LLM
LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr"
}

@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client and embeddings model.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        # Use a fixed directory instead of a temporary one
        db_path = Path("./chroma_db")
        db_path.mkdir(exist_ok=True)
        
        db_client = chromadb.PersistentClient(path=str(db_path))
        embeddings_model = TogetherEmbeddings(
            together_api_key=TOGETHER_API_KEY,
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        return db_client, embeddings_model
    except Exception as e:
        st.error(f"An error occurred during dependency initialization: {e}.")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=st.session_state.embeddings_model
    )

def clear_chroma_data():
    """Clears all data from the ChromaDB collection and the local directory."""
    try:
        # Delete the collection from the database
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
        
        # Also, remove the physical directory
        db_path = Path("./chroma_db")
        if db_path.exists() and db_path.is_dir():
            shutil.rmtree(db_path)
        
        st.toast("Chat data and database cleared!", icon="🧹")

    except Exception as e:
        st.error(f"Error clearing collection or directory: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def is_valid_github_raw_url(url):
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def process_and_store_documents(documents):
    """
    Processes a list of text documents and stores them in ChromaDB.
    """
    collection = get_collection()
    
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        ids=document_ids
    )
    
    st.toast("Documents processed and stored successfully!", icon="✅")

# --- Agentic RAG Pipeline ---
class RAGQuery(BaseModel):
    """Input for RAG queries."""
    query: str = Field(description="The user's question or request.")

@tool
def retrieve_from_documents(query: str) -> List[str]:
    """
    Retrieves the most relevant documents from the ChromaDB collection based on a query.
    Returns a list of document chunks as strings.
    """
    collection = get_collection()
    if collection.count() == 0:
        return ["No documents found in the database. Please upload a file first."]
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    return results['documents'][0]

def setup_agentic_rag():
    """
    Sets up the LangChain agent for Agentic RAG.
    """
    llm = Together(
        together_api_key=TOGETHER_API_KEY,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=2048
    )
    
    tools = [retrieve_from_documents]
    
    prompt = hub.pull("hwchase17/react")
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

def run_agentic_pipeline(query, selected_language):
    """
    Executes the full Agentic RAG pipeline.
    """
    if 'agent_executor' not in st.session_state:
        st.session_state.agent_executor = setup_agentic_rag()
    
    final_prompt = f"The user asks: '{query}'. Provide the response in {selected_language}. If you can't find an answer from the provided documents, state that you cannot answer."
    
    try:
        response = st.session_state.agent_executor.invoke({"input": final_prompt})
        return response.get("output", "An error occurred or no output was generated.")
    except Exception as e:
        return f"An error occurred during the agent execution: {e}"

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your document..."):
        try:
            collection = get_collection()
            if collection.count() == 0:
                st.error("Please process a document first by uploading a file or entering a URL.")
                return
        except Exception as e:
            st.error(f"Error checking document count: {e}")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_agentic_pipeline(prompt, st.session_state.selected_language)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(layout="wide")

    # Sidebar
    with st.sidebar:
        st.header("Agentic RAG Flow")
        st.session_state.selected_language = st.selectbox(
            "Select a Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        if st.button("New Chat"):
            st.session_state.messages = []
            clear_chroma_data()
            st.session_state.chat_history = {}
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
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()

    # Main content area
    st.title("Agentic RAG Chat Flow")
    st.markdown("---")
    
    # Initialize dependencies outside of the main UI block to prevent re-initialization
    if 'db_client' not in st.session_state or 'embeddings_model' not in st.session_state:
        st.session_state.db_client, st.session_state.embeddings_model = initialize_dependencies()

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

    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
