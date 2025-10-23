import streamlit as st
import os
import tempfile
import uuid
import requests
import re
from datetime import datetime
from typing import List

# import google.generativeai as genai # Not strictly needed if only using LangChain components
# from langchain_tavily import TavilySearch # Removed as requested
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

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
    """Initializes ChromaDB client and SentenceTransformer model."""
    try:
        # Use a local temporary directory for ChromaDB to ensure persistence during a session
        # and cleanup upon session restart/app stop.
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        # SentenceTransformer for embedding, running on CPU
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"Dependency initialization error: {e}")
        st.stop()

if 'db_client' not in st.session_state or 'model' not in st.session_state:
    st.session_state.db_client, st.session_state.model = initialize_dependencies()

def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

@tool
def retrieve_documents(query: str) -> str:
    """
    Searches the ChromaDB vector database for documents relevant to the query.
    """
    try:
        collection = get_collection()
        model = st.session_state.model
        # Encode the query using the model
        query_embedding = model.encode(query).tolist()
        # Query the collection
        results = collection.query(query_embeddings=query_embedding, n_results=5)
        
        # Check if documents were found and return them
        if results and results.get('documents') and results['documents'][0]:
            return "\n---\n".join(results['documents'][0])
        else:
            return "No relevant documents found in the database."
            
    except Exception as e:
        return f"Error in document retrieval: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression string safely.
    """
    # Simple check to prevent complex code execution via eval
    if not re.match(r"^[0-9+\-*/().\s]+$", expression):
        return "Invalid expression. Only basic arithmetic operations are allowed."
    try:
        # Use a limited global/local environment for safety, though still 'eval'
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def duckduckgo_search(query: str) -> str:
    """
    Performs a DuckDuckGo web search for the given query. Use for general knowledge questions.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

def create_agent():
    """Creates and returns the LangChain ReAct agent executor."""
    prompt_template = hub.pull("hwchase17/react-chat")
    
    # --- Tools ---
    # TavilySearch tool removed as requested
    tools = [
        retrieve_documents, # For RAG
        calculator,       # For math
        duckduckgo_search   # For general web search
    ]

    # --- Gemini LLM Setup ---
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets.")
        st.stop()
    
    # Use Gemini Flash 3B model explicitly as requested (or fall back to a common flash model)
    # The 'gemini-2.5-flash' is the currently recommended name for high-speed chat.
    gemini_model_name = st.secrets.get("GEMINI_MODEL_NAME", "gemini-2.5-flash") 

    llm = ChatGoogleGenerativeAI(
        google_api_key=gemini_api_key,
        model=gemini_model_name,
        temperature=0.1
    )
    
    # --- Agent Creation ---
    agent = create_react_agent(llm, tools, prompt_template)
    # Note: AgentExecutor now requires `handle_parsing_errors=True` for robust chat interaction
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def clear_chroma_data():
    """Clears all documents from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100) -> List[str]:
    """Splits a large text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents: List[str]):
    """Embeds and stores documents in ChromaDB."""
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=document_ids)
    st.toast("Documents processed and stored successfully!", icon="✅")

def is_valid_github_raw_url(url: str) -> bool:
    """Validates if the URL is a raw GitHub file with .txt or .md extension."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)$"
    return re.match(pattern, url) is not None

def display_chat_messages():
    """Displays the current chat history in the main area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles user input and gets a response from the agent."""
    if prompt := st.chat_input("Ask about your document or a general question..."):
        # 1. Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_executor = create_agent()
                
                # Format chat history for the agent to maintain context
                # LangChain expects a list of tuples (role, content) for chat_history in ReAct
                history_for_agent = []
                for msg in st.session_state.messages[:-1]: # Exclude the current user message
                    history_for_agent.append((msg["role"], msg["content"]))
                
                try:
                    response = agent_executor.invoke({
                        "input": prompt, 
                        "chat_history": history_for_agent # Pass the formatted history
                    })
                    final_response = response.get('output', 'An error occurred during agent execution.')
                except Exception as e:
                    final_response = f"An error occurred: {e}"
                
                st.markdown(final_response)
                
        # 3. Store assistant message and update chat history for the session
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        
        # Auto-update chat title on first message
        if st.session_state.current_chat_id:
            chat_data = st.session_state.chat_history.get(st.session_state.current_chat_id)
            if chat_data and chat_data['title'] == "New Chat":
                # Use the first 5 words of the prompt as a simple title
                new_title = " ".join(prompt.split()[:5]) + "..." if len(prompt.split()) > 5 else prompt
                st.session_state.chat_history[st.session_state.current_chat_id]['title'] = new_title

# --- Main UI ---
st.set_page_config(page_title="Agentic RAG Chat with Gemini Flash")
st.title("Agentic RAG Chat Flow with Gemini Flash Model")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Chat Controls")
    
    # Button for New Chat
    if st.button("Start New Chat"):
        st.session_state.messages = []
        clear_chroma_data() # Clear RAG data for a new session
        # Create a new chat ID and initial history entry
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_history[new_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }
        st.experimental_rerun()
        
    st.subheader("Chat History")
    # Display and manage chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        sorted_chat_ids = sorted(
            st.session_state.chat_history.keys(),
            key=lambda x: st.session_state.chat_history[x]['date'],
            reverse=True
        )
        for chat_id in sorted_chat_ids:
            chat_data = st.session_state.chat_history[chat_id]
            chat_title = chat_data['title']
            date_str = chat_data['date'].strftime("%b %d, %I:%M %p")
            
            # Highlight the current chat
            is_current = chat_id == st.session_state.current_chat_id
            button_label = f"**{'* ' if is_current else ''}{chat_title}{'*' if is_current else ''}** - {date_str}"
            
            if st.button(button_label, key=f"hist_btn_{chat_id}"):
                # Switch to a different chat
                if st.session_state.current_chat_id != chat_id:
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()
    
# Document Upload and Processing Area
with st.container():
    st.subheader("Add Context Documents")
    uploaded_files = st.file_uploader("Upload text files (.txt)", type="txt", accept_multiple_files=True)
    github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:", key="github_url_input")

    # Handle file uploads
    if uploaded_files:
        if st.button("Process Uploaded Files", key="process_files_btn"):
            with st.spinner("Processing files..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    file_contents = uploaded_file.read().decode("utf-8")
                    documents = split_documents(file_contents)
                    all_documents.extend(documents)
                
                if all_documents:
                    process_and_store_documents(all_documents)
                    st.success("All files processed and stored successfully! You can now ask questions about their content.")
                else:
                    st.warning("No content found in the uploaded files to process.")

    # Handle GitHub URL
    if github_url:
        if is_valid_github_raw_url(github_url):
            if st.button("Process URL", key="process_url_btn"):
                with st.spinner("Fetching and processing file from URL..."):
                    try:
                        response = requests.get(github_url)
                        response.raise_for_status() # Raise HTTPError for bad responses
                        file_contents = response.text
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                        st.success("File from URL processed! You can now chat about its contents.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching URL: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        else:
            st.warning("Please ensure the URL is a raw GitHub link ending in `.txt` or `.md`.")


# Main Chat Display and Input
display_chat_messages()
handle_user_input()
