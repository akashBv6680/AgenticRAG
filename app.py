import streamlit as st
import os
import tempfile
import uuid
import requests
import re
from datetime import datetime
from typing import List, TypedDict, Annotated, Sequence
import operator

# Core dependencies
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain/Google GenAI imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph, END
# Note: Using a simple dictionary for in-memory checkpointing in Streamlit
# For production persistence, use a LangGraph Checkpoint backend (e.g., SQLite, Postgres)

COLLECTION_NAME = "agentic_rag_documents"

# --- State Initialization ---
# Ensure all session state keys are correctly initialized for Streamlit Cloud deployment
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_session.chat_history[st.session_session.current_chat_id] = {
        'messages': st.session_session.messages,
        'title': "New Chat",
        'date': datetime.now()
    }

@st.cache_resource
def initialize_dependencies():
    """Initializes ChromaDB client and SentenceTransformer model."""
    try:
        # Use a temporary directory for ChromaDB storage during the session
        db_path = tempfile.mkdtemp() 
        db_client = chromadb.PersistentClient(path=db_path)
        # SentenceTransformer for embedding, running on CPU
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"Dependency initialization error: {e}")
        st.stop()

# Initialize resources
if 'db_client' not in st.session_session or 'model' not in st.session_session:
    st.session_session.db_client, st.session_session.model = initialize_dependencies()

def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_session.db_client.get_or_create_collection(name=COLLECTION_NAME)

# --- Tool Definitions ---
@tool
def retrieve_documents(query: str) -> str:
    """
    Searches the ChromaDB vector database for documents relevant to the query (Internal RAG).
    Use this tool ONLY to answer questions about the uploaded files or processed URLs.
    """
    try:
        collection = get_collection()
        model = st.session_session.model
        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3) 
        
        if results and results.get('documents') and results['documents'][0]:
            return "\n---\n".join(results['documents'][0])
        else:
            return "No relevant documents found in the database."
            
    except Exception as e:
        return f"Error in document retrieval: {e}"

@tool
def duckduckgo_search(query: str) -> str:
    """
    Performs a DuckDuckGo web search for the given query (External Knowledge).
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

# --- LangGraph Setup ---

# Define the State
class GraphState(TypedDict):
    """Represents the state of our graph/conversation."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    rag_context: str
    web_context: str
    next_tool: str 
    
def get_llm():
    """Initializes and returns the ChatGoogleGenerativeAI model."""
    # Securely retrieve the key from Streamlit secrets
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it for deployment.")
        st.stop()
    
    gemini_model_name = st.secrets.get("GEMINI_MODEL_NAME", "gemini-2.5-flash") 

    # Bind tools to the LLM for function calling capability (CAG Step 1: Process Input/Decide)
    llm = ChatGoogleGenerativeAI(
        google_api_key=gemini_api_key,
        model=gemini_model_name,
        temperature=0.0
    ).bind_tools(tools=[retrieve_documents, duckduckgo_search])
    return llm

# --- LangGraph Nodes ---

def call_model_to_decide(state: GraphState):
    """Decides which tool (if any) to use based on the query and history."""
    llm = get_llm()
    
    system_prompt = (
        "You are an expert Context-Augmented Generation (CAG) system. "
        "Your task is to decide whether to use 'retrieve_documents' (internal RAG), "
        "'duckduckgo_search' (external web search), or directly provide a 'Final Answer'. "
        "Prioritize the RAG tool if the question is about the uploaded content."
    )
    
    response = llm.invoke([HumanMessage(content=system_prompt)] + state["messages"])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        return {"messages": state["messages"] + [response], "next_tool": tool_name}
    else:
        return {"messages": state["messages"] + [response], "next_tool": "Final Answer"}

def call_rag_tool(state: GraphState):
    """Executes the retrieve_documents tool."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    rag_result = retrieve_documents.invoke({"query": query})
    
    # Inject context into the state (CAG Step 3, 4, 6)
    return {"rag_context": rag_result, "messages": state["messages"]}

def call_web_tool(state: GraphState):
    """Executes the duckduckgo_search tool."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    web_result = duckduckgo_search.invoke({"query": query})
    
    # Inject context into the state (CAG Step 3, 5, 6)
    return {"web_context": web_result, "messages": state["messages"]}

def generate_final_response(state: GraphState):
    """Generates the final, context-augmented response (CAG Steps 7-12)."""
    llm = get_llm()
    
    rag_context = state.get("rag_context", "No internal documents retrieved.")
    web_context = state.get("web_context", "No web search performed.")
    
    final_prompt = (
        "You are a sophisticated, context-aware AI assistant. "
        "Provide a final, complete, and coherent answer using the provided context and conversation history. "
        "**CAG Principles:** 1. Prioritize **RAG Context**. 2. **Augment** with **Web Context** if needed. "
        "3. **Check Consistency and Align Context** with the user's prior messages.\n\n"
        f"--- RAG Context ---\n{rag_context}\n"
        f"--- Web Context ---\n{web_context}\n"
        f"--- User Messages/History Follow ---\n"
    )
    
    messages_with_context = [HumanMessage(content=final_prompt)] + state["messages"]
    response = llm.invoke(messages_with_context)
    
    return {"messages": state["messages"] + [response]}

# --- Graph Flow Logic ---

def route_tools(state: GraphState):
    """Decides the next step based on the LLM's decision."""
    next_tool = state.get("next_tool")
    
    if next_tool == "retrieve_documents":
        return "call_rag"
    elif next_tool == "duckduckgo_search":
        return "call_web"
    elif next_tool == "Final Answer":
        return END 
    else:
        # If a tool was called and executed, the next step is final generation
        return "generate_response"
        
# Build the Graph
@st.cache_resource
def get_graph():
    """Initializes and compiles the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    workflow.add_node("decide_tool", call_model_to_decide)
    workflow.add_node("call_rag", call_rag_tool)
    workflow.add_node("call_web", call_web_tool)
    workflow.add_node("generate_response", generate_final_response)

    workflow.set_entry_point("decide_tool")

    # Conditional routing after the decision node
    workflow.add_conditional_edges(
        "decide_tool",
        route_tools,
        {"call_rag": "call_rag", "call_web": "call_web", "end": END},
    )

    # After tool calls, go to final generation
    workflow.add_edge("call_rag", "generate_response")
    workflow.add_edge("call_web", "generate_response")

    # After final generation, the flow is complete
    workflow.add_edge("generate_response", END)

    return workflow.compile()

# --- Utility Functions (RAG/Data Processing) ---

def clear_chroma_data():
    """Clears all documents from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_session.db_client.list_collections()]:
            st.session_session.db_client.delete_collection(name=COLLECTION_NAME)
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
    model = st.session_session.model

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
    for message in st.session_session.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Main Interaction Logic (Streaming) ---

def handle_user_input():
    """Handles user input and gets a streamed response from the LangGraph agent."""
    if prompt := st.chat_input("Ask about your document or a general question..."):
        # 1. Store user message
        st.session_session.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Setup Graph and initial state
        graph_app = get_graph()
        
        # Convert session messages to LangChain BaseMessage objects
        lc_messages = []
        for msg in st.session_session.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Initial state for LangGraph
        initial_state = {
            "messages": lc_messages,
            "question": prompt,
            "rag_context": "",
            "web_context": "",
            "next_tool": "",
        }

        # 3. Get agent response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the graph execution
                for s in graph_app.stream(initial_state):
                    # Only stream the final generation node's output
                    if s.get("generate_response"):
                        latest_message_chunk = s["generate_response"].get("messages", [])[-1]
                        
                        if isinstance(latest_message_chunk, AIMessage):
                            full_response += latest_message_chunk.content or ""
                            response_placeholder.markdown(full_response + "▌") # Streaming effect
                            
                    # Handle the case where the answer is END'd directly from decision (no tool needed)
                    elif s.get("decide_tool") and s.get("decide_tool").get("next_tool") == "Final Answer":
                        # The full message is in the state's messages list
                        final_message = s.get("decide_tool")["messages"][-1]
                        full_response = final_message.content
                        break # Exit loop since it's the final answer
                
                response_placeholder.markdown(full_response) # Final content without cursor

            except Exception as e:
                full_response = f"An error occurred in the agent: {e}"
                response_placeholder.markdown(full_response)
                
        # 4. Store assistant message and update chat history
        st.session_session.messages.append({"role": "assistant", "content": full_response})
        
        # Auto-update chat title on first message
        if st.session_session.current_chat_id:
            chat_data = st.session_session.chat_history.get(st.session_session.current_chat_id)
            if chat_data and chat_data['title'] == "New Chat":
                new_title = " ".join(prompt.split()[:5]) + "..." if len(prompt.split()) > 5 else prompt
                st.session_session.chat_history[st.session_session.current_chat_id]['title'] = new_title


# --- Main UI ---
st.set_page_config(page_title="LangGraph CAG-RAG Chat with Gemini Flash")
st.title("LangGraph Context-Augmented RAG (CAG) with Gemini Flash 🚀")
st.markdown("This agent uses a state machine to intelligently apply RAG (internal documents) and Web Search (external knowledge) based on the query and conversation history, fulfilling your CAG requirements.")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("Start New Chat"):
        st.session_session.messages = []
        clear_chroma_data() # Clear RAG context for the new chat
        new_chat_id = str(uuid.uuid4())
        st.session_session.current_chat_id = new_chat_id
        st.session_session.chat_history[new_chat_id] = {
            'messages': st.session_session.messages,
            'title': "New Chat",
            'date': datetime.now()
        }
        st.experimental_rerun()
        
    st.subheader("Chat History")
    
    if 'chat_history' in st.session_session and st.session_session.chat_history:
        sorted_chat_ids = sorted(
            st.session_session.chat_history.keys(),
            key=lambda x: st.session_session.chat_history[x]['date'],
            reverse=True
        )
        for chat_id in sorted_chat_ids:
            chat_data = st.session_session.chat_history[chat_id]
            chat_title = chat_data['title']
            date_str = chat_data['date'].strftime("%b %d, %I:%M %p")
            
            is_current = chat_id == st.session_session.current_chat_id
            button_label = f"**{'* ' if is_current else ''}{chat_title}{'*' if is_current else ''}** - {date_str}"
            
            if st.button(button_label, key=f"hist_btn_{chat_id}"):
                if st.session_session.current_chat_id != chat_id:
                    st.session_session.current_chat_id = chat_id
                    st.session_session.messages = st.session_session.chat_history[chat_id]['messages']
                    st.experimental_rerun()
    
# Document Upload and Processing Area
with st.container():
    st.subheader("Add Context Documents (RAG Knowledge Base) 📚")
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
                    st.success("Files processed! The agent will prioritize RAG for related questions.")
                else:
                    st.warning("No content found in the uploaded files to process.")

    # Handle GitHub URL
    if github_url:
        if is_valid_github_raw_url(github_url):
            if st.button("Process URL", key="process_url_btn"):
                with st.spinner("Fetching and processing file from URL..."):
                    try:
                        response = requests.get(github_url)
                        response.raise_for_status()
                        file_contents = response.text
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                        st.success("File from URL processed! The agent will prioritize RAG for related questions.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching URL: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        else:
            st.warning("Please ensure the URL is a raw GitHub link ending in `.txt` or `.md`.")


# Main Chat Display and Input
display_chat_messages()
handle_user_input()
