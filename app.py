# app.py

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS   # ✅ updated import
from pypdf import PdfReader   # ✅ updated import

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM (updated model)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- Streamlit UI ---
st.title("📚 Chatbot with Gemini + PDF Support")

# --- Normal Chat ---
st.subheader("💬 Chat with Gemini")
user_input = st.text_input("You:", "")
if user_input:
    response = llm.invoke(user_input)
    st.write("🤖:", response.content)

# --- PDF Q&A ---
st.subheader("📑 Upload a PDF to chat with it")
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    st.success("✅ PDF processed! Ask me questions below ⬇️")

    pdf_question = st.text_input("Ask about the PDF:", "")
    if pdf_question:
        result = qa_chain.invoke({"question": pdf_question, "chat_history": []})
        st.write("🤖:", result["answer"])
