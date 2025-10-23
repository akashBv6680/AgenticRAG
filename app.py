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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

# --- FIX 1: CORRECTED IMPORT PATH FOR ToolException ---
from langchain_core.tools import ToolException 
# ----------------------------------------------------

from langchain_community.tools import DuckDuckGoSearchRun

# --- FIX 3: CORRECTED IMPORT NAME FOR GOOGLE CHAT MODEL ---
from langchain_google_genai import ChatGoogleGenerativeAI # <- FIXED: Added 'tive'
# ----------------------------------------------------------

from langchain_community.document_loaders import TextLoader, WebBaseLoader

# LangGraph imports
from langgraph.graph import StateGraph, END

COLLECTION_NAME = "agentic_rag_documents"

# --- FIX 2: CORRECTED st.session_state REFERENCES ---
# --- State Initialization ---
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
# Initial check to ensure the current chat exists in history
if st.session_state.current_chat_id not in st.session_state.chat_history:
     st.session_state.chat_history[st.session_state.current_chat_id] = {
        'messages': st.session_state.messages,
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

if 'db_client' not in st.session_state or 'model' not in st.session_state:
    st.session_state.db_client, st.session_state.model = initialize_dependencies()

def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

# --- Tool Definitions ---
@tool
def retrieve_documents(query: str) -> str:
    """
    Searches the ChromaDB vector database for documents relevant to the query (Internal RAG).
    Use this tool ONLY to answer questions about the uploaded files or processed URLs.
    """
    try:
        collection = get_collection()
        model = st.session_state.model
        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3) 
        
        if results and results.get('documents') and results['documents'][0]:
            return "\n---\n".join(results['documents'][0])
        else:
            return "No relevant documents found in the internal RAG database."
            
    except Exception as e:
        st.warning(f"RAG Tool Error: {e}") 
        # Raise ToolException so the graph can correctly handle the error and inform the LLM
        raise ToolException(f"Error in document retrieval: {e}. Cannot use RAG context.")

@tool
def duckduckgo_search(query: str) -> str:
    """
    Performs a DuckDuckGo web search for the given query (External Knowledge).
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        st.warning(f"Web Search Tool Error: {e}")
        # Raise ToolException so the graph can correctly handle the error and inform the LLM
        raise ToolException(f"Error in web search: {e}. Cannot use Web context.")

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
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it for deployment.")
        st.stop()
    
    gemini_model_name = st.secrets.get("GEMINI_MODEL_NAME", "gemini-2.5-flash") 

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
    
    # Pass the full message history, prepended by the system prompt
    response = llm.invoke([HumanMessage(content=system_prompt)] + state["messages"])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        
        # Add the model's tool call message to the history
        return {"messages": [response], "next_tool": tool_name}
    else:
        # Add the model's final response message to the history
        return {"messages": [response], "next_tool": "Final Answer"}

def call_rag_tool(state: GraphState):
    """Executes the retrieve_documents tool."""
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return {"rag_context": "Tool call failed to parse.", "messages": []}
        
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    tool_call_id = tool_call["id"]
    
    try:
        rag_result = retrieve_documents.invoke({"query": query})
        
        tool_message = ToolMessage(
            content=rag_result, 
            tool_call_id=tool_call_id
        )
    except ToolException as e:
        # Catch the exception raised by the tool and create a ToolMessage with the error
        tool_message = ToolMessage(
            content=f"Tool Execution Error: {str(e)}", 
            tool_call_id=tool_call_id
        )
        rag_result = f"Error: {str(e)}"
    
    return {"rag_context": rag_result, "messages": [tool_message]}

def call_web_tool(state: GraphState):
    """Executes the duckduckgo_search tool."""
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return {"web_context": "Tool call failed to parse.", "messages": []}
        
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    tool_call_id = tool_call["id"]
    
    try:
        web_result = duckduckgo_search.invoke({"query": query})
        
        tool_message = ToolMessage(
            content=web_result, 
            tool_call_id=tool_call_id
        )
    except ToolException as e:
        # Catch the exception raised by the tool and create a ToolMessage with the error
        tool_message = ToolMessage(
            content=f"Tool Execution Error: {str(e)}", 
            tool_call_id=tool_call_id
        )
        web_result = f"Error: {str(e)}"
    
    return {"web_context": web_result, "messages": [tool_message]}

def generate_final_response(state: GraphState):
    """Generates the final, context-augmented response (CAG Steps 7-12)."""
    llm = get_llm()
    
    rag_context = state.get("rag_context", "No internal documents retrieved.")
    web_context = state.get("web_context", "No web search performed.")
    
    final_prompt_template = (
        "You are a sophisticated, context-aware AI assistant. "
        "Provide a final, complete, and coherent answer using the provided context and conversation history. "
        "**CAG Principles:** 1. Prioritize **RAG Context**. 2. **Augment** with **Web Context** if needed. "
        "3. **Check Consistency and Align Context** with the user's prior messages. "
        "If you have executed tools, use the provided context to answer the user's last question.\n\n"
        f"--- RAG Context ---\n{rag_context}\n"
        f"--- Web Context ---\n{web_context}\n"
        f"--- Messages/History Follow ---\n"
    )
    
    messages_with_context = [HumanMessage(content=final_prompt_template)] + state["messages"]
    
    try:
        response = llm.invoke(messages_with_context)
    except Exception as e:
        error_message = f"LLM Generation Error: The model failed to generate a response. ({e})"
        st.error(error_message)
        response = AIMessage(content=error_message)
        
    return {"messages": [response]} 

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
        return END
        
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

    workflow.add_conditional_edges(
        "decide_tool",
        route_tools,
        {"call_rag": "call_rag", "call_web": "call_web", "Final Answer": END},
    )

    workflow.add_edge("call_rag", "generate_response")
    workflow.add_edge("call_web", "generate_response")

    workflow.add_edge("generate_response", END)

    return workflow.compile()

# --- Utility Functions (Data Processing) ---

def clear_chroma_data():
    """Clears all documents from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
            get_collection() 
            st.toast("RAG database cleared.", icon="🗑️")
        else:
            st.toast("RAG database is already empty.", icon="🤷")
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
    if not documents:
        st.warning("No documents to process.")
        return
        
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=document_ids)
    st.toast(f"Processed {len(documents)} document chunks and stored them in the RAG database.", icon="✅")

def is_valid_github_raw_url(url: str) -> bool:
    """Validates if the URL is a raw GitHub file with .txt or .md extension."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)$"
    return re.match(pattern, url) is not None

def display_chat_messages():
    """Displays the current chat history in the main area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Main Interaction Logic (Streaming) ---

def handle_user_input():
    """Handles user input and gets a streamed response from the LangGraph agent."""
    if prompt := st.chat_input("Ask about your document or a general question..."):
        # 1. Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Setup Graph and initial state
        graph_app = get_graph()
        
        # Convert session messages to LangChain BaseMessage objects
        lc_messages = []
        for msg in st.session_state.messages:
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
            
            # --- START DIAGNOSTIC SECTION ---
            st.info("Agent Debug Log (Tracking LangGraph Flow):")
            current_log = st.empty()
            log_messages = []
            
            try:
                for i, s in enumerate(graph_app.stream(initial_state)):
                    # Get the node that just executed
                    node_name = list(s.keys())[0]
                    node_output = s[node_name]
                    
                    log_messages.append(f"Step {i+1}: Node **{node_name}** executed.")
                    
                    if node_name == "decide_tool":
                        decision = node_output.get('next_tool', 'Unknown')
                        log_messages.append(f"   -> Decision: **{decision}**")
                        current_log.markdown("\n".join(log_messages))
                        
                        if decision == "Final Answer":
                            final_message = node_output["messages"][-1]
                            full_response = final_message.content
                            response_placeholder.markdown(full_response)
                            break 
                        
                    elif node_name == "call_rag":
                        rag_result = node_output.get('rag_context', 'N/A')
                        log_messages.append(f"   -> RAG Context Status: {'SUCCESS' if not rag_result.startswith('Error') else 'FAILURE'}")
                        current_log.markdown("\n".join(log_messages))
                        
                    elif node_name == "call_web":
                        web_result = node_output.get('web_context', 'N/A')
                        log_messages.append(f"   -> Web Context Status: {'SUCCESS' if not web_result.startswith('Error') else 'FAILURE'}")
                        current_log.markdown("\n".join(log_messages))

                    elif node_name == "generate_response":
                        latest_message_chunk = node_output.get("messages", [])[-1]
                        
                        if isinstance(latest_message_chunk, AIMessage):
                            full_response += latest_message_chunk.content or ""
                            response_placeholder.markdown(full_response + "▌") 
                        
                response_placeholder.markdown(full_response) 

            except Exception as e:
                full_response = f"An **CRITICAL** error occurred in the agent execution: `{e}`. Check the debug log above for the last successful step."
                response_placeholder.markdown(full_response)
                log_messages.append(f"**Execution Failed with Error:** `{e}`")
                current_log.markdown("\n".join(log_messages))
                
        # 4. Store assistant message and update chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Auto-update chat title on first message
        if st.session_state.current_chat_id:
            chat_data = st.session_state.chat_history.get(st.session_state.current_chat_id)
            if chat_data and chat_data['title'] == "New Chat":
                new_title = " ".join(prompt.split()[:5]) + "..." if len(prompt.split()) > 5 else prompt
                st.session_state.chat_history[st.session_state.current_chat_id]['title'] = new_title


# --- Main UI ---
st.set_page_config(page_title="LangGraph CAG-RAG Chat with Gemini Flash")
st.title("LangGraph Context-Augmented RAG (CAG) with Gemini Flash 🚀")
st.markdown("This agent uses a state machine to intelligently apply **RAG (internal documents)** and **Web Search (external knowledge)** based on the query and conversation history.")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("1. RAG Data Setup")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a .txt file:", type=['txt'])
    if uploaded_file is not None:
        try:
            string_data = uploaded_file.getvalue().decode("utf-8")
            
            if st.button(f"Process and Embed File: {uploaded_file.name}", key="upload_btn"):
                st.info("Splitting document and creating embeddings...")
                text_chunks = split_documents(string_data)
                process_and_store_documents(text_chunks)
                st.success(f"File '{uploaded_file.name}' processed.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # URL Input
    github_url = st.text_input("Or, enter a Raw GitHub URL (.txt or .md):", key="url_input")
    if st.button("Process and Embed URL", key="url_btn"):
        if github_url and is_valid_github_raw_url(github_url):
            try:
                st.info("Fetching content from URL...")
                loader = WebBaseLoader(github_url)
                docs = loader.load()
                
                if docs and docs[0].page_content:
                    st.info("Splitting document and creating embeddings...")
                    text_chunks = split_documents(docs[0].page_content)
                    process_and_store_documents(text_chunks)
                    st.success(f"URL '{github_url}' content processed.")
                else:
                    st.warning("Could not fetch content from the URL.")
            except Exception as e:
                st.error(f"Error fetching/processing URL: {e}")
        elif github_url:
            st.error("Invalid raw GitHub URL format. Must end with .txt or .md.")
            
    # Clear Data Button
    st.markdown("---")
    st.header("2. Chat Controls")
    if st.button("Start New Chat & Clear RAG Data", type="primary"):
        st.session_state.messages = [] 
        clear_chroma_data() 
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_history[new_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }
        st.experimental_rerun()
        
    st.subheader("3. Chat History")
    
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
            
            is_current = chat_id == st.session_state.current_chat_id
            button_label = f"**{'* ' if is_current else ''}{chat_title}{'*' if is_current else ''}** - {date_str}"
            
            if st.button(button_label, key=f"hist_btn_{chat_id}"):
                if st.session_state.current_chat_id != chat_id:
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()
    
# Main Chat Display and Input
display_chat_messages()
handle_user_input()
