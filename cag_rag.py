import streamlit as st
import psutil
from langchain.document_loaders import PDFPlumberLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important; color: #E0E0E0 !important; border-radius: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important; color: #F0F0F0 !important; border-radius: 10px;
    }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model='deepseek-r1:1.5b')
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_document(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_processor.split_documents(raw_documents)

def index_document(documents_chunk):
    DOCUMENT_VECTOR_DB.add_documents(documents_chunk)

def find_related_document(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_document):
    context_text = "\n\n".join([doc.page_content for doc in context_document])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def get_memory_status():
    memory_info = psutil.virtual_memory()
    return f"Memory Usage: {memory_info.percent}% (Used: {memory_info.used / (1024 ** 3):.2f} GB / Total: {memory_info.total / (1024 ** 3):.2f} GB)"

st.title("📘 NeuroDoc AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf", help="Select a PDF document for analysis")

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_document(raw_docs)
    index_document(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    st.text(get_memory_status())
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_document(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
        
        st.text(get_memory_status())
