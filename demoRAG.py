import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Free RAG PDF Chatbot (HuggingFace + FAISS)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


if uploaded_file:

    # Save uploaded PDF
    with open("C:\FAQ_UseCase\RAG_Implementation_Guide.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader("C:\FAQ_UseCase\RAG_Implementation_Guide.pdf")
    docs = loader.load()

    # -----------------------------
    # Split into chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # -----------------------------
    # Embeddings (Free)
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Vector Store (FAISS)
    # -----------------------------
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # -----------------------------
    # Free HuggingFace LLM
    # -----------------------------
    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # -----------------------------
    # Prompt Template
    # -----------------------------
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    # -----------------------------
    # RAG Chain
    # -----------------------------
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -----------------------------
    # User Question
    # -----------------------------
    query = st.text_input("Ask a question from the PDF:")

    if query:
        response = rag_chain.invoke(query)

        st.subheader("Answer:")
        st.success(response)
