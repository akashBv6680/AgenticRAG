import streamlit as st
import random
import time
import json
import base64
from typing import TypedDict, Annotated, List, Union, Dict, Any
import operator
import os

# --- Firebase Imports (Mandatory for Canvas Environment) ---
from firebase_admin import initialize_app, firestore, credentials
from google.cloud.firestore import Client as FirestoreClient # Type hint for Firestore

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# --- Utility Functions for Firebase/Auth (MUST BE INCLUDED) ---
def init_firebase():
    """Initializes Firebase app and returns Firestore and Auth clients."""
    if 'db' not in st.session_state:
        try:
            # 1. Load Config
            firebase_config = json.loads(__firebase_config)
            
            # 2. Initialize Firebase Admin SDK (used for configuration)
            cred = credentials.Certificate(firebase_config)
            initialize_app(cred, name=__app_id) # Use __app_id for unique app name
            
            # 3. Get Firestore Client
            db = firestore.client(app=initialize_app.get_app(__app_id))

            # Store the initialized clients in session state
            st.session_state.db = db
            st.session_state.app_id = __app_id
            st.session_state.user_id = "canvas_user" # Simplified userId for demo purposes

            # Logging initialization status
            print("Firebase successfully initialized.")
        except Exception as e:
            # Fallback/Error handling
            print(f"Error initializing Firebase: {e}")
            st.session_state.db = None
            st.error("Failed to initialize Firebase. Persistence will not work.")

# --- LangGraph State Definition ---
# This dictionary defines the state that passes between nodes in the graph
class AgentState(TypedDict):
    # messages is a list that will be appended to by each node
    messages: Annotated[List[BaseMessage], operator.add]
    # A single source of truth for the entire final response
    final_response: str
    # A list of tool calls made in the current step
    tool_calls: List[dict]
    # Flag to signal if the RAG tool should be used
    use_rag: bool

# --- Tool Definitions (MOCK/SIMPLIFIED FOR RUNNABILITY) ---

@tool
def google_search(query: str) -> str:
    """
    Use this tool to search the web for real-time information.
    Always use this for questions about current events, news, or general knowledge.
    """
    st.toast("Searching the web...")
    # Mocking a search result
    if "Gemini" in query:
        return "Search result: Gemini 2.5 Flash is highly optimized for low-latency tasks and is often preferred for chat applications to enhance speed."
    elif "Streamlit" in query:
        return "Search result: Streamlit is a Python library that lets you create and share web apps for data science and machine learning."
    else:
        return f"Search result: I found general information about '{query}'. For faster response, streaming is recommended."

@tool
def retrieve_documents(query: str) -> str:
    """
    Use this tool ONLY to look up information from the loaded internal documents (RAG).
    """
    # In a real app, this would perform embedding, ChromaDB query, and return chunks.
    st.toast("Retrieving documents from vector store...")
    if st.session_state.get('rag_enabled', False):
         return f"RAG document context: The document states that the main bottleneck in agent applications is the multiple sequential LLM calls required for reasoning and tool execution. Streaming can mask this latency."
    else:
        return "RAG document context: No specific internal documents are loaded or relevant for this query."
    
# Gather all tools
tools = [google_search, retrieve_documents]

# --- LangGraph Node Definitions ---

def call_model(state: AgentState) -> dict:
    """Invokes the chat model to determine the next action (tool call or final answer)."""
    
    # 1. Extract the current conversation history
    # We only send the last few messages to prevent token bloat
    history_limit = 10 
    history = state["messages"][-history_limit:]

    # 2. Get the LLM bound with tools
    llm = st.session_state.llm_agent
    
    try:
        # 3. Invoke the model
        response = llm.invoke(history)
    except Exception as e:
        # Handle API errors gracefully
        return {"messages": [AIMessage(content=f"Error contacting the model: {e}")], "final_response": "Error"}

    # 4. Check for tool calls
    if response.tool_calls:
        # Pass the tool calls to the next node (call_tool)
        return {"messages": [response], "tool_calls": response.tool_calls}
    else:
        # This is the final answer, ready to be streamed/displayed
        return {"messages": [response], "final_response": response.content}

def call_tool(state: AgentState) -> dict:
    """Executes the tool calls requested by the model."""
    
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Execute all tool calls
    tool_executor = st.session_state.tool_executor
    tool_outputs = tool_executor.invoke(last_message)
    
    # 2. Format tool outputs into ToolMessages
    tool_messages = [
        ToolMessage(content=json.dumps(output["output"]), tool_call_id=output["id"])
        for output in tool_outputs
    ]
    
    # 3. Return the tool results to the graph state
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """Conditional edge to decide next step: call tool, or finish."""
    
    # Check if the last message contains tool calls (meaning the model wants to use a tool)
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    
    # Check if the LLM has already generated the final response content
    if state.get("final_response"):
        return "end"

    # If neither, continue reasoning (should not happen in this simple graph, but is safe)
    return "call_model"
    
# --- LangGraph Initialization ---

def get_or_create_langgraph(tools, system_prompt):
    """Initializes and compiles the LangGraph."""
    if 'graph' in st.session_state:
        return st.session_state.graph

    # Initialize LLM with the specified system prompt and tools
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools=tools)

    # 1. Create the ToolExecutor
    tool_executor = ToolExecutor(tools)

    # Store LLM and ToolExecutor for use in nodes
    st.session_state.llm_agent = llm_with_tools
    st.session_state.tool_executor = tool_executor
    
    # 2. Define the Graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (steps in the process)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)
    
    # Set the entry point
    workflow.set_entry_point("call_model")
    
    # Define edges (transitions between nodes)
    # The start always goes to the model
    
    # Conditional edge after calling the model
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tool": "call_tool", # If tool is needed, call tool
            "end": END,                # If final answer is ready, finish
            "call_model": "call_model" # Loop back if needed (e.g. error recovery)
        }
    )

    # After calling the tool, we loop back to the model for final synthesis
    workflow.add_edge("call_tool", "call_model")
    
    # 3. Compile the graph
    graph = workflow.compile()
    st.session_state.graph = graph
    return graph

# --- Streamlit UI and Logic ---

def handle_user_input():
    """Processes the user's message and streams the response from the agent."""
    prompt = st.session_state.prompt_input
    if not prompt:
        return

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.prompt_input = "" # Clear input

    # Prepare chat history for the agent
    history_for_agent = []
    # Reverse the history to get the latest messages first, and limit it
    for msg in st.session_state.messages[-10:]:
         if msg["role"] == "user":
             history_for_agent.append(HumanMessage(content=msg["content"]))
         elif msg["role"] == "assistant":
             history_for_agent.append(AIMessage(content=msg["content"]))

    # Prepare input for the graph (HumanMessage for the current prompt)
    # The graph expects a list of messages for the initial state.
    inputs = {"messages": [HumanMessage(content=prompt)]}

    # ----------------------------------------------------
    # CORE CHANGE: LangGraph Streaming Logic
    # ----------------------------------------------------
    
    # We create a placeholder to update the text in real-time
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Get the initialized graph
        graph = st.session_state.graph

        try:
            # Stream the graph execution
            # The streaming will yield state updates for each node transition
            for chunk in graph.stream(inputs):
                # LangGraph state updates often contain the new message list.
                # We check for the last message in the list.
                
                # Check for updates to the messages list
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    
                    # We are only interested in the final AIMessage content
                    if isinstance(last_message, AIMessage):
                        # Use .content to get the text chunk for streaming
                        text_chunk = last_message.content
                        
                        if text_chunk:
                            # Append and update the placeholder
                            full_response += text_chunk
                            # Add a subtle blinking cursor to indicate streaming
                            response_placeholder.markdown(full_response + "▌") 
                            
            # Final update without the cursor
            response_placeholder.markdown(full_response)
            
            # Add the complete response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            full_response = "Sorry, I ran into an error while processing that request."
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    # Set Streamlit Page Config
    st.set_page_config(page_title="LangGraph Streaming Agent", layout="wide")

    # 1. Initialize Firebase/Auth (Crucial for Canvas persistence)
    init_firebase()
    
    # 2. Define System Prompt and Initialize LangGraph
    SYSTEM_PROMPT = (
        "You are a helpful and extremely fast AI assistant powered by Gemini Flash and LangGraph. "
        "Your primary goal is to provide concise and accurate answers. "
        "Use the tools provided ONLY when necessary to answer the question. "
        "Prioritize the 'retrieve_documents' tool for internal knowledge first, then 'google_search' for current events or external knowledge. "
        "Always respond directly and conversationally."
    )
    
    # Caching the graph creation to avoid re-initializing on every rerun
    try:
        if 'graph' not in st.session_state:
            get_or_create_langgraph(tools, SYSTEM_PROMPT)
    except Exception as e:
        st.error(f"Failed to initialize the LangGraph. Check API key setup. Error: {e}")
        return

    # 3. Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hello! I am a Gemini-powered Agent. Ask me anything, and notice the rapid streaming response thanks to LangGraph!"}
        )
    
    # 4. Sidebar and UI Setup
    st.sidebar.title("Agent Configuration")
    st.sidebar.markdown(f"**App ID:** `{st.session_state.app_id}`")
    st.sidebar.markdown(f"**User ID:** `{st.session_state.user_id}`")
    st.sidebar.markdown("---")
    
    rag_enabled = st.sidebar.checkbox("Enable RAG Tool (Mocked)", value=True, help="Simulates using internal document retrieval.")
    st.session_state.rag_enabled = rag_enabled
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # 5. Main Chat Interface
    st.title("⚡ LangGraph Streaming Agent (Gemini Flash)")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(
