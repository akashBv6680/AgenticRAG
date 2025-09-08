import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime, timedelta
import re
import shutil

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
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain import hub
from langchain_community.llms import Together
from sentence_transformers import SentenceTransformer


# --- Constants and Configuration ---
COLLECTION_NAME = "agentic_rag_documents"
# TOGETHER_API_KEY should be set in Streamlit Secrets
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_together_api_key_here")
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
if 'db_client' not in st.session_state or 'model' not in st.session_state:
    @st.cache_resource
    def initialize_dependencies():
        """
        Initializes and returns the ChromaDB client and SentenceTransformer model.
        Using @st.cache_resource ensures this runs only once.
        """
        try:
            db_path = tempfile.mkdtemp()
            db_client = chromadb.PersistentClient(path=db_path)
            # Use SentenceTransformer for embedding, as it's more lightweight
            model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            return db_client, model
        except Exception as e:
            st.error(f"An error occurred during dependency initialization: {e}.")
            st.stop()

    st.session_state.db_client, st.session_state.model = initialize_dependencies()


def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    # Note: SentenceTransformer is not a ChromaDB-native embedding function, so we will use it
    # manually to embed and then store/query.
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )


@tool
def retrieve_documents(query: str) -> str:
    """Searches for and returns documents relevant to the query from the vector database."""
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


def call_together_api(prompt, max_retries=5):
    """
    Calls the Together AI API with exponential backoff for retries.
    """
    # The call_together_api function from your previous code
    retry_delay = 1
    for i in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
            if e.response.status_code == 429:
                time.sleep(retry_delay)
                retry_delay *= 2
            elif e.response.status_code == 401:
                st.error("Invalid API Key. Please check your Together AI API key.")
                return {"error": "401 Unauthorized"}
            else:
                st.error(f"Failed to call API after {i+1} retries: {e}")
                return {"error": str(e)}
        except Exception as e:
            st.error(f"An error occurred during the API call: {e}")
            return {"error": str(e)}


def create_agent():
    """Creates and returns a LangChain agent executor."""
    prompt_template = hub.pull("hwchase17/react-chat")
    tools = [retrieve_documents]
    together_llm = Together(
        together_api_key=TOGETHER_API_KEY,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    agent = create_tool_calling_agent(together_llm, tools, prompt_template)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")


def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)


def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    # Generate embeddings for the documents using the SentenceTransformer model
    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

    st.toast("Documents processed and stored successfully!", icon="✅")


def is_valid_github_raw_url(url):
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
                    response = agent_executor.invoke({"input": prompt})
                    final_response = response.get('output', 'An error occurred.')
                except Exception as e:
                    final_response = f"An error occurred: {e}"
                st.markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})


# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(layout="wide")

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

    # Main content area
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
                        st.error(f"An unexpected
