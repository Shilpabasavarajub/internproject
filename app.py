import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

st.set_page_config(
    page_title="AI Document Copilot",
    layout="wide"
)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:right; padding-right:40px;">
        AI Document Copilot
    </h1>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UPLOAD CARD
# -----------------------------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container(border=True):
            st.markdown("### Upload File")
            uploaded_file = st.file_uploader(
                "Drag and drop files here",
                type=["pdf"],
                label_visibility="collapsed"
            )

            build_btn = st.button(
                "Build Knowledge Base",
                use_container_width=True
            )

# -----------------------------
# FILE CHIP DISPLAY
# -----------------------------
if uploaded_file:
    file_path = f"data/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown(
        f"""
        <div style="
            width:60%;
            margin:auto;
            padding:10px;
            border-radius:8px;
            background-color:#f3f4f6;
            font-size:14px;
        ">
            📄 {uploaded_file.name}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# BUILD KNOWLEDGE BASE
# -----------------------------
if uploaded_file and build_btn:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    chroma_client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        anonymized_telemetry=False
    )
)


    vectordb = Chroma.from_documents(
    docs,
    embedding=embeddings,
    client=chroma_client
)

    vectordb.persist()

    st.success("✅ Knowledge base created")

# -----------------------------
# ACTION CARDS
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.container(border=True)
    st.markdown("📄 **Summarize this PDF**")

with c2:
    st.container(border=True)
    st.markdown("❓ **Explain these files**")

with c3:
    st.container(border=True)
    st.markdown("💬 **Ask questions**")

# -----------------------------
# CHAT INPUT (BOTTOM)
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

query = st.chat_input("Ask about your documents...")

if query:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2
    )

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(response.content)
