import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
import uuid
import re

# -----------------------------
# Load API keys
# -----------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Page layout
# -----------------------------
st.set_page_config(page_title="Chat Bubble RAG Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Modern RAG Chatbot")
st.markdown("Ask questions, see live answers, and view clickable references from the websites you added.")

# -----------------------------
# Sidebar: Settings & website management
# -----------------------------
st.sidebar.header("Settings")
chunk_size = st.sidebar.number_input("Chunk size:", value=2500, step=500)
chunk_overlap = st.sidebar.number_input("Chunk overlap:", value=250, step=50)
show_history = st.sidebar.checkbox("Show Chat History")

st.sidebar.markdown("---")
st.header("Website Management")
if "urls" not in st.session_state:
    st.session_state["urls"] = []

# Add new website
url_input = st.text_input("Enter website URL:", placeholder="https://example.com")
if st.button("Add Website") and url_input:
    st.session_state["urls"].append(url_input)
    st.success(f"Website added: {url_input}")

# Display added websites with remove buttons
if st.session_state["urls"]:
    st.subheader("Added Websites")
    for i, url in enumerate(st.session_state["urls"]):
        cols = st.columns([4, 1])
        cols[0].markdown(f'{i+1}. [🔗 {url}]({url})', unsafe_allow_html=True)
        if cols[1].button("❌", key=f"remove_{i}"):
            st.session_state["urls"].pop(i)
            st.success(f"Website removed: {url}")
            
            # Safe rerun by updating a dummy state variable
            if "rerun_trigger" not in st.session_state:
                st.session_state["rerun_trigger"] = 0
            st.session_state["rerun_trigger"] += 1

# -----------------------------
# Clear chat history
# -----------------------------

if "confirm_clear" not in st.session_state:
    st.session_state["confirm_clear"] = False

if st.sidebar.button("🗑️ Clear Chat History") and not st.session_state["confirm_clear"]:
    st.session_state["confirm_clear"] = True

if st.session_state["confirm_clear"]:
    st.sidebar.warning("⚠️ This will delete all chat history permanently!")
    if st.sidebar.button("Confirm Clear History"):
        st.session_state["chat_history"] = []
        st.session_state["pending_chat"] = {}
        st.sidebar.success("Chat history cleared!")
        st.session_state["confirm_clear"] = False
    if st.sidebar.button("Cancel"):
        st.session_state["confirm_clear"] = False

# -----------------------------
# Session state
# -----------------------------
if "docs" not in st.session_state:
    st.session_state["docs"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------------
# Load & split documents
# -----------------------------
@st.cache_data(show_spinner=True)
def load_docs(urls, chunk_size=500, chunk_overlap=100):
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_docs.extend(splitter.split_documents(docs))
    return all_docs

# -----------------------------
# Create vectorstore
# -----------------------------
@st.cache_resource(show_spinner=True)
def create_vectorstore(docs):
    vect = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
    return vect.as_retriever()

# -----------------------------
# Initialize LLM
# -----------------------------
llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0,
    streaming=True
)

# -----------------------------
# Custom callback for streaming
# -----------------------------
class StreamlitTokenStreamer(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

# -----------------------------
# Custom RAG prompt
# -----------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Mention the sources used.
Question: {question} 
Context: {context} 
Answer:

"""
)

# -----------------------------
# Build RAG chain
# -----------------------------
rag_chain = None
if st.session_state["urls"]:
    with st.spinner("Loading websites and creating chatbot..."):
        st.session_state["docs"] = load_docs(st.session_state["urls"], chunk_size, chunk_overlap)
        retriever = create_vectorstore(st.session_state["docs"])
        retriever.search_kwargs['k'] = 3
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )
    st.success("Knowledge base ready! You can now chat.")

# -----------------------------
# Temporary container for current answer
# -----------------------------
current_answer_container = st.empty()

# -----------------------------
# Show question input
# -----------------------------
if st.session_state["urls"]:
    with st.form("query_form"):
        query = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Submit")

    if submitted and query and rag_chain:
        answer_id = str(uuid.uuid4())
        callback_handler = StreamlitTokenStreamer(current_answer_container)

        # Display user bubble
        st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

        # Stream bot answer
        result = rag_chain.invoke({"query": query}, callbacks=[callback_handler])
        answer_text = result["result"]

        # -----------------------------
        # Map answer to sources actually used in answer
        # -----------------------------
        sources_mapping = []
        for doc in result.get("source_documents", []):
            src_url = doc.metadata.get("source", "No source")
            overlap = sum(1 for word in doc.page_content.split() if word in answer_text.split())
            if overlap > 0:
                sources_mapping.append({"url": src_url, "text_snippet": doc.page_content.strip()})

        # -----------------------------
        # Display bot answer
        # -----------------------------
        st.markdown(f'<div class="bot-bubble">{answer_text}</div>', unsafe_allow_html=True)

        # -----------------------------
        # Display only sources actually used
        # -----------------------------
        if sources_mapping:
            st.markdown("**Sources:**")
            for src in sources_mapping:
                st.markdown(f'- 🔗 [View Source]({src["url"]})', unsafe_allow_html=True)

        # Copy answer button
        if st.button("📋 Copy Answer", key=f"copy_answer_{answer_id}"):
            st.text_area("Copy the answer:", value=answer_text, height=150)

        # Save chat history
        if st.session_state.get("pending_chat"):
            st.session_state["chat_history"].append(st.session_state["pending_chat"])
        st.session_state["pending_chat"] = {
            "user": query,
            "bot": answer_text,
            "sources": sources_mapping,
            "answer_id": answer_id
        }
else:
    st.info("Please add at least one website to enable asking questions.")

# -----------------------------
# Optional chat history
# -----------------------------
if show_history and st.session_state["chat_history"]:
    st.subheader("Chat History")
    for chat in st.session_state["chat_history"]:
        st.markdown(f'<div class="user-bubble">{chat["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-bubble">{chat["bot"]}</div>', unsafe_allow_html=True)
        if chat.get("sources"):
            st.markdown("**Sources:**")
            for src in chat["sources"]:
                st.markdown(f'- 🔗 [View Source]({src["url"]})', unsafe_allow_html=True)
