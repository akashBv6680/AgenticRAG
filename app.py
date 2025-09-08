import streamlit as st
import os
import sys
import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Together
import tempfile
import json
import requests
import re
import time

# --- Pysqlite3 fix for Streamlit Cloud ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed.")
    st.stop()

# --- Initialize Dependencies ---
@st.cache_resource
def initialize_dependencies():
    db_path = tempfile.mkdtemp()
    db_client = Chroma(persist_directory=db_path)
    # Using LangChain's wrapper for the embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return db_client, embedding_function

st.session_state.db_client, st.session_state.embedding_function = initialize_dependencies()

# --- Tools and Agent Setup ---
@tool
def retrieve_documents(query: str) -> str:
    """Searches for and returns documents relevant to the query from the vector database."""
    docs = st.session_state.db_client.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in docs])

# Define the LLM for the agent
together_llm = Together(
    together_api_key=os.environ.get("TOGETHER_API_KEY"),
    model="mistralai/Mistral-7B-Instruct-v0.2"
)

# Agent setup
prompt_template = hub.pull("hwchase17/react-chat")
tools = [retrieve_documents]
agent = create_tool_calling_agent(together_llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Document Processing ---
def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents):
    st.session_state.db_client.add_texts(
        texts=documents,
        embedding=st.session_state.embedding_function
    )
    st.toast("Documents processed and stored!", icon="✅")

# --- Streamlit UI ---
def main_ui():
    st.set_page_config(layout="wide")
    st.title("Agentic RAG Chatflow")
    st.markdown("---")

    # Document upload section
    with st.container():
        st.subheader("Add Context Documents")
        uploaded_files = st.file_uploader("Upload text files (.txt)", type="txt", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    for uploaded_file in uploaded_files:
                        file_contents = uploaded_file.read().decode("utf-8")
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                    st.success("All files processed and stored successfully!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the input to the agent executor
                response = agent_executor.invoke({"input": prompt})
                # The agent's final answer is in a specific key
                final_response = response.get('output', 'An error occurred.')
                st.markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main_ui()
