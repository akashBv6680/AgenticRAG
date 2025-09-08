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
        ids=document_
