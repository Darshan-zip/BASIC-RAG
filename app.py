import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Futuristic UI Styling
st.set_page_config(page_title="RAG System", page_icon="📄", layout="wide")

st.title("RAG System: Document Intelligence")
st.markdown("---")

# Configuration
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
DB_DIR = "./chroma_db"

# Initialize Components
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for Uploads
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    process_btn = st.button("Process Document")

if uploaded_file and process_btn:
    with st.status("Processing document...", expanded=True) as status:
        # Save file locally for PyPDFLoader
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.write("⏳ Extracting Text...")
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        
        st.write("⏳ Chunking Data...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        st.write("⏳ Embedding & Storing in ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        st.session_state.vector_store = vector_store
        status.update(label="Processing Complete!", state="complete")
        st.success("Document indexed and ready for querying.")

# Main Query Interface
st.subheader("💬 Query Documents")
query = st.text_input("Enter your query across the uploaded knowledge base:")

if query:
    if st.session_state.vector_store is None:
        st.error("Please upload and process a document first!")
    else:
        with st.spinner("Retrieving answer..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever()
            )
            response = qa_chain.invoke(query)
            st.markdown(f"### 🤖 Response\n{response['result']}")
