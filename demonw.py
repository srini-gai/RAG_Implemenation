import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("🤖 ChatGPT-Style PDF RAG Chatbot (Free)")
st.write("Upload a PDF and chat with it like ChatGPT 💬")


# -------------------------------
# Session State: Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file and st.session_state.retriever is None:

    with open("C:\FAQ_UseCase\RAG_Implementation_Guide.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully! Processing...")

    # Load PDF
    loader = PyPDFLoader("C:\FAQ_UseCase\RAG_Implementation_Guide.pdf")
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Free LLM (GPT-Neo)
    hf_pipeline = pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-125M",
        max_new_tokens=150,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Save to session
    st.session_state.retriever = retriever
    st.session_state.llm = llm

    st.success("✅ RAG Chatbot Ready! Ask questions below.")


# -------------------------------
# Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------
# Chat Input Box
# -------------------------------
if st.session_state.retriever:

    user_query = st.chat_input("Ask something from the PDF...")

    if user_query:

        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve context
        retrieved_docs = st.session_state.retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Build prompt
        prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If not found, say: "I cannot find the answer in the document."

Context:
{context_text}

Question:
{user_query}

Answer:
"""

        # Generate response
        response = st.session_state.llm.invoke(prompt)

        # Show assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)

else:
    st.info("👆 Upload a PDF to start chatting.")
