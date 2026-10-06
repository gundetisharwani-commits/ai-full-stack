import os
import re
import hashlib
import sqlite3
from datetime import datetime

import streamlit as st
import chromadb
import ollama
import pandas as pd
import plotly.express as px
import fitz

from sentence_transformers import SentenceTransformer
from docx import Document
from pptx import Presentation


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "EduNexus"
LLM_MODEL = "gemma3:1b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
DATABASE = os.path.join(BASE_DIR, "edunexus.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


st.set_page_config(
    page_title="EduNexus",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f3f4f6;
    }

    .brand {
        padding: 10px 5px 25px 5px;
    }

    .brand-title {
        font-size: 27px;
        font-weight: 800;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
    }

    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #111827,
            #243b68
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }

    .hero p {
        color: #dbeafe;
        font-size: 15px;
        margin-top: 8px;
    }

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.05);
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
    }

    .card-text {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.6;
    }

    .source-card {
        background: #f8fafc;
        border-left: 4px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 13px;
    }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }

    .metric-label {
        font-size: 12px;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_document" not in st.session_state:
    st.session_state.selected_document = "All Materials"


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE)

    # Create table if it does not exist
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_type TEXT,
            topic TEXT,
            document TEXT,
            created_at TEXT
        )
        """
    )

    # Check existing columns
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(activity)"
        ).fetchall()
    ]

    # Add missing columns to old database
    if "topic" not in columns:
        connection.execute(
            "ALTER TABLE activity ADD COLUMN topic TEXT"
        )

    if "document" not in columns:
        connection.execute(
            "ALTER TABLE activity ADD COLUMN document TEXT"
        )

    if "created_at" not in columns:
        connection.execute(
            "ALTER TABLE activity ADD COLUMN created_at TEXT"
        )

    connection.commit()
    connection.close()


def save_activity(activity_type, topic, document):

    connection = sqlite3.connect(DATABASE)

    connection.execute(
        """
        INSERT INTO activity
        (activity_type, topic, document, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            activity_type,
            topic,
            document,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


def get_activity():

    connection = sqlite3.connect(DATABASE)

    dataframe = pd.read_sql_query(
        "SELECT * FROM activity ORDER BY id DESC",
        connection
    )

    connection.close()

    return dataframe


initialize_database()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_vector_database():

    client = chromadb.PersistentClient(
        path=VECTOR_DIR
    )

    collection = client.get_or_create_collection(
        name="edunexus"
    )

    return collection


collection = load_vector_database()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DOCUMENT LOADERS
# ============================================================

def read_pdf(path):

    pages = []

    with fitz.open(path) as pdf:

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            text = clean_text(
                page.get_text()
            )

            if text:

                pages.append(
                    {
                        "text": text,
                        "page": page_number
                    }
                )

    return pages


def read_docx(path):

    document = Document(path)

    text = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text.append(
                paragraph.text.strip()
            )

    complete_text = "\n".join(text)

    if complete_text.strip():

        return [
            {
                "text": complete_text,
                "page": 1
            }
        ]

    return []


def read_pptx(path):

    presentation = Presentation(path)

    slides = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        texts = []

        for shape in slide.shapes:

            if hasattr(shape, "text"):

                if shape.text.strip():

                    texts.append(
                        shape.text.strip()
                    )

        slide_text = "\n".join(texts)

        if slide_text.strip():

            slides.append(
                {
                    "text": slide_text,
                    "page": slide_number
                }
            )

    return slides


def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    if text.strip():

        return [
            {
                "text": text,
                "page": 1
            }
        ]

    return []


def load_document(path):

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension == ".pdf":
        return read_pdf(path)

    if extension == ".docx":
        return read_docx(path)

    if extension == ".pptx":
        return read_pptx(path)

    if extension == ".txt":
        return read_txt(path)

    raise ValueError(
        "Only PDF, DOCX, PPTX and TXT files are supported."
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=700,
    overlap=100
):

    chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            ).strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"]
                    }
                )

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def create_document_id(
    filename,
    content
):

    file_hash = hashlib.md5(
        content
    ).hexdigest()[:10]

    name = os.path.splitext(
        filename
    )[0]

    name = re.sub(
        r"[^a-zA-Z0-9]",
        "_",
        name
    )

    return f"{name}_{file_hash}"


def index_file(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    document_id = create_document_id(
        uploaded_file.name,
        file_bytes
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(
        save_path,
        "wb"
    ) as file:

        file.write(file_bytes)

    pages = load_document(
        save_path
    )

    chunks = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No readable text was found in this document."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = []

    metadatas = []

    for number, chunk in enumerate(
        chunks
    ):

        ids.append(
            f"{document_id}_{number}"
        )

        metadatas.append(
            {
                "document": uploaded_file.name,
                "page": chunk["page"],
                "document_id": document_id
            }
        )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# ============================================================
# GET UPLOADED DOCUMENTS
# ============================================================

def get_uploaded_documents():

    if collection.count() == 0:
        return []

    data = collection.get(
        include=["metadatas"]
    )

    documents = set()

    for metadata in data["metadatas"]:

        if metadata:

            document = metadata.get(
                "document"
            )

            if document:
                documents.add(document)

    return sorted(
        documents
    )


# ============================================================
# RAG SEARCH
# ============================================================

def retrieve_information(
    question,
    selected_document,
    number_of_results=6
):

    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    parameters = {
        "query_embeddings": [
            query_embedding
        ],
        "n_results": min(
            number_of_results,
            collection.count()
        ),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if (
        selected_document
        and
        selected_document != "All Materials"
    ):

        parameters["where"] = {
            "document": selected_document
        }

    try:

        result = collection.query(
            **parameters
        )

    except Exception:

        return []

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    sources = []

    for index, text in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        sources.append(
            {
                "text": text,
                "document": metadata.get(
                    "document",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "?"
                )
            }
        )

    return sources


# ============================================================
# CONTEXT
# ============================================================

def create_context(
    sources,
    maximum_characters=10000
):

    context = []

    total = 0

    for source in sources:

        block = (
            f"SOURCE: {source['document']} "
            f"| PAGE: {source['page']}\n"
            f"{source['text']}"
        )

        if (
            total + len(block)
            >
            maximum_characters
        ):
            break

        context.append(
            block
        )

        total += len(block)

    return "\n\n".join(
        context
    )


# ============================================================
# OLLAMA
# ============================================================

def generate_ai_response(
    instruction,
    context
):

    system_message = """
You are EduNexus, an educational AI assistant.

IMPORTANT:
Use ONLY the study material provided in the context.

Do not use outside knowledge.

If the requested information is not present
in the uploaded material, say:

"I couldn't find this information in the uploaded material."

Never invent facts.

Give clear, organized and student-friendly answers.
"""

    user_message = f"""
UPLOADED STUDY MATERIAL:

{context}

TASK:

{instruction}
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return response[
            "message"
        ][
            "content"
        ]

    except Exception as error:

        return (
            "⚠️ Ollama error.\n\n"
            f"{error}\n\n"
            "Make sure Ollama is running and "
            f"{LLM_MODEL} is installed."
        )


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(
    sources
):

    if not sources:
        return

    with st.expander(
        "📚 View information used from your document"
    ):

        for source in sources:

            st.markdown(
                f"""
                <div class="source-card">
                <b>📄 {source['document']}</b>
                &nbsp; | &nbsp;
                Page/Slide: {source['page']}
                <br><br>
                {source['text'][:600]}...
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DOCUMENT SELECTOR
# ============================================================

def document_selector():

    documents = get_uploaded_documents()

    if not documents:

        st.warning(
            "📚 Please upload and process a study document first."
        )

        return None

    options = [
        "All Materials"
    ] + documents

    if (
        st.session_state.selected_document
        not in options
    ):

        st.session_state.selected_document = (
            "All Materials"
        )

    selected = st.selectbox(
        "📚 Knowledge Source",
        options,
        index=options.index(
            st.session_state.selected_document
        )
    )

    st.session_state.selected_document = selected

    return selected


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">
                🎓 EduNexus
            </div>

            <div class="brand-subtitle">
                AI-Powered Learning Workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### WORKSPACE")

    menu = st.radio(
        "",
        [
            "🏠 Dashboard",
            "📚 Knowledge Library",
            "🤖 AI Tutor",
            "📝 Smart Notes",
            "🎯 Exam Preparation",
            "❓ Practice Quiz",
            "🃏 Flashcards",
            "🧠 Concept Explorer",
            "📅 Study Planner",
            "📊 Learning Analytics"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    documents = get_uploaded_documents()

    st.caption(
        f"📚 {len(documents)} document(s) indexed"
    )

    if documents:

        sidebar_document = st.selectbox(
            "Current material",
            [
                "All Materials"
            ] + documents
        )

        st.session_state.selected_document = (
            sidebar_document
        )

    else:

        st.info(
            "Upload your study material "
            "to activate the AI features."
        )


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Welcome to EduNexus 🎓
        </h1>

        <p>
        Your personal AI learning workspace —
        powered by your own study materials.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    documents = get_uploaded_documents()

    activity = get_activity()

    col1, col2, col3, col4 = st.columns(4)

    values = [
        (
            col1,
            len(documents),
            "Documents"
        ),
        (
            col2,
            len(activity),
            "Activities"
        ),
        (
            col3,
            len(
                activity[
                    activity["activity_type"]
                    == "AI Tutor"
                ]
            ) if not activity.empty else 0,
            "Tutor Questions"
        ),
        (
            col4,
            len(
                activity[
                    activity["activity_type"]
                    == "Smart Notes"
                ]
            ) if not activity.empty else 0,
            "Notes Created"
        )
    ]

    for column, number, label in values:

        with column:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-number">
                {number}
                </div>

                <div class="metric-label">
                {label}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### What can EduNexus do?")

    col1, col2, col3 = st.columns(3)

    features = [
        (
            col1,
            "🤖 AI Tutor",
            "Ask questions from your uploaded material."
        ),
        (
            col2,
            "📝 Smart Notes",
            "Generate revision notes from your material."
        ),
        (
            col3,
            "🧠 Concept Explorer",
            "Understand difficult topics."
        )
    ]

    for column, title, description in features:

        with column:

            st.markdown(
                f"""
                <div class="card">

                <div class="card-title">
                {title}
                </div>

                <div class="card-text">
                {description}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# KNOWLEDGE LIBRARY
# ============================================================

elif menu == "📚 Knowledge Library":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Knowledge Library 📚
        </h1>

        <p>
        Upload the documents EduNexus should learn from.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload study material",
        type=[
            "pdf",
            "docx",
            "pptx",
            "txt"
        ],
        accept_multiple_files=True
    )

    if st.button(
        "🚀 Process & Add to Knowledge Base",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please select at least one file."
            )

        else:

            progress = st.progress(0)

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                try:

                    chunk_count = index_file(
                        uploaded_file
                    )

                    save_activity(
                        "Document Upload",
                        uploaded_file.name,
                        uploaded_file.name
                    )

                    st.success(
                        f"✅ {uploaded_file.name} "
                        f"processed successfully — "
                        f"{chunk_count} chunks."
                    )

                except Exception as error:

                    st.error(
                        f"❌ {uploaded_file.name}: "
                        f"{error}"
                    )

                progress.progress(
                    (index + 1)
                    /
                    len(uploaded_files)
                )

    st.markdown("### Indexed Documents")

    documents = get_uploaded_documents()

    if documents:

        for document in documents:

            st.markdown(
                f"""
                <div class="card">

                📄 <b>{document}</b>

                <br>

                <span class="card-text">
                Available as knowledge for
                AI Tutor, Notes, Quiz, Flashcards,
                Exam Preparation and Concept Explorer.
                </span>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No documents uploaded yet."
        )


# ============================================================
# AI TUTOR
# ============================================================

elif menu == "🤖 AI Tutor":

    st.markdown(
        """
        <div class="hero">

        <h1>
        AI Tutor 🤖
        </h1>

        <p>
        Ask anything from your uploaded study material.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    documents = get_uploaded_documents()

    if not documents:

        st.warning(
            "📚 Go to Knowledge Library and upload "
            "your study material first."
        )

    else:

        document_selector()

        st.markdown(
            "### 💬 Ask your question"
        )

        # Display previous messages
        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # THIS IS THE IMPORTANT PART:
        # chat_input is outside forms and receives cursor focus.
        question = st.chat_input(
            "Type your question here..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            sources = retrieve_information(
                question,
                st.session_state.selected_document
            )

            with st.chat_message(
                "assistant"
            ):

                if not sources:

                    answer = (
                        "I couldn't find this information "
                        "in your uploaded material."
                    )

                    st.warning(answer)

                else:

                    context = create_context(
                        sources
                    )

                    with st.spinner(
                        "Searching your material..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Answer this question:

                            {question}

                            Explain clearly for a student.
                            Use ONLY the supplied material.
                            """,
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            save_activity(
                "AI Tutor",
                question,
                st.session_state.selected_document
            )


# ============================================================
# SMART NOTES
# ============================================================

elif menu == "📝 Smart Notes":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Smart Notes 📝
        </h1>

        <p>
        Generate revision notes directly from your uploaded material.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if document_selector():

        topic = st.text_input(
            "Enter topic or chapter",
            placeholder="Example: Fundamental Rights"
        )

        if st.button(
            "✨ Generate Smart Notes",
            type="primary",
            use_container_width=True
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                sources = retrieve_information(
                    topic,
                    st.session_state.selected_document
                )

                if not sources:

                    st.error(
                        "This topic was not found in "
                        "your uploaded material."
                    )

                else:

                    context = create_context(
                        sources
                    )

                    with st.spinner(
                        "Creating notes from your material..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Create smart revision notes
                            for the topic:

                            {topic}

                            Include:
                            • Definition
                            • Important points
                            • Key concepts
                            • Examples mentioned in the material
                            • Quick revision summary

                            Use ONLY the uploaded material.
                            """,
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

                    save_activity(
                        "Smart Notes",
                        topic,
                        st.session_state.selected_document
                    )


# ============================================================
# EXAM PREPARATION
# ============================================================

elif menu == "🎯 Exam Preparation":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Exam Preparation 🎯
        </h1>

        <p>
        Create exam-ready answers from your study material.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if document_selector():

        topic = st.text_input(
            "Topic",
            placeholder="Example: Fundamental Rights"
        )

        answer_type = st.selectbox(
            "Answer type",
            [
                "2-Mark Answer",
                "5-Mark Answer",
                "10-Mark Answer"
            ]
        )

        if st.button(
            "Generate Answer",
            type="primary",
            use_container_width=True
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                sources = retrieve_information(
                    topic,
                    st.session_state.selected_document
                )

                if not sources:

                    st.error(
                        "Topic not found in uploaded material."
                    )

                else:

                    context = create_context(
                        sources
                    )

                    instructions = {

                        "2-Mark Answer":
                        f"""
                        Give a concise 2-mark answer
                        about {topic}.
                        """,

                        "5-Mark Answer":
                        f"""
                        Give a structured 5-mark answer
                        about {topic}.
                        Include introduction,
                        main points and conclusion.
                        """,

                        "10-Mark Answer":
                        f"""
                        Give a detailed 10-mark answer
                        about {topic}.
                        Include introduction,
                        headings, important points,
                        examples if available,
                        and conclusion.
                        """
                    }

                    with st.spinner(
                        "Preparing exam answer..."
                    ):

                        answer = generate_ai_response(
                            instructions[
                                answer_type
                            ],
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

                    save_activity(
                        "Exam Preparation",
                        topic,
                        st.session_state.selected_document
                    )


# ============================================================
# QUIZ
# ============================================================

elif menu == "❓ Practice Quiz":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Practice Quiz ❓
        </h1>

        <p>
        Test your knowledge using questions generated from your material.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if document_selector():

        topic = st.text_input(
            "Quiz topic",
            placeholder="Example: Indian Constitution"
        )

        number = st.slider(
            "Number of questions",
            5,
            15,
            10
        )

        if st.button(
            "Generate Quiz",
            type="primary",
            use_container_width=True
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                sources = retrieve_information(
                    topic,
                    st.session_state.selected_document
                )

                if not sources:

                    st.error(
                        "Topic not found in uploaded material."
                    )

                else:

                    context = create_context(
                        sources
                    )

                    with st.spinner(
                        "Generating quiz..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Create {number} MCQs about
                            {topic}.

                            Format each question like:

                            Question:
                            A.
                            B.
                            C.
                            D.
                            Correct Answer:
                            Explanation:

                            Use ONLY the uploaded material.
                            """,
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

                    save_activity(
                        "Practice Quiz",
                        topic,
                        st.session_state.selected_document
                    )


# ============================================================
# FLASHCARDS
# ============================================================

elif menu == "🃏 Flashcards":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Flashcards 🃏
        </h1>

        <p>
        Turn important information into quick revision cards.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if document_selector():

        topic = st.text_input(
            "Flashcard topic",
            placeholder="Example: Directive Principles"
        )

        number = st.slider(
            "Number of cards",
            5,
            20,
            10
        )

        if st.button(
            "Generate Flashcards",
            type="primary",
            use_container_width=True
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                sources = retrieve_information(
                    topic,
                    st.session_state.selected_document
                )

                if not sources:

                    st.error(
                        "Topic not found in uploaded material."
                    )

                else:

                    context = create_context(
                        sources
                    )

                    with st.spinner(
                        "Creating flashcards..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Create {number} flashcards
                            about {topic}.

                            Format:

                            Card 1
                            Front: question
                            Back: answer

                            Keep them short and useful
                            for revision.

                            Use ONLY the uploaded material.
                            """,
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

                    save_activity(
                        "Flashcards",
                        topic,
                        st.session_state.selected_document
                    )


# ============================================================
# CONCEPT EXPLORER
# ============================================================

elif menu == "🧠 Concept Explorer":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Concept Explorer 🧠
        </h1>

        <p>
        Understand difficult concepts using your study material.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if document_selector():

        topic = st.text_input(
            "Concept",
            placeholder="Example: Separation of Powers"
        )

        if st.button(
            "Explore Concept",
            type="primary",
            use_container_width=True
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a concept."
                )

            else:

                sources = retrieve_information(
                    topic,
                    st.session_state.selected_document
                )

                if not sources:

                    st.error(
                        "Concept not found in uploaded material."
                    )

                else:

                    context = create_context(
                        sources
                    )

                    with st.spinner(
                        "Exploring concept..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Explain the concept:

                            {topic}

                            Include:
                            • Simple meaning
                            • Main idea
                            • Important points
                            • Related concepts in the material
                            • Examples from the material
                            • Quick revision points

                            Use ONLY the uploaded material.
                            """,
                            context
                        )

                    st.markdown(answer)

                    display_sources(
                        sources
                    )

                    save_activity(
                        "Concept Explorer",
                        topic,
                        st.session_state.selected_document
                    )


# ============================================================
# STUDY PLANNER
# ============================================================
# ============================================================
# STUDY PLANNER
# ============================================================
elif menu == "📅 Study Planner":

    # --------------------------------------------------------
    # STUDY PLANNER HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="hero">

            <h1>📅 AI Study Planner</h1>

            <p>
            Turn your uploaded study material into a
            personalized learning schedule.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    documents = get_uploaded_documents()

    if not documents:

        st.warning(
            "📚 Please upload and process your study material "
            "from Knowledge Library first."
        )

    else:

        # ----------------------------------------------------
        # SELECT KNOWLEDGE SOURCE
        # ----------------------------------------------------

        selected_document = document_selector()

        if selected_document:

            st.markdown(
                "### 🎯 Build Your Study Plan"
            )

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                    Customize your learning plan
                    </div>

                    <div class="card-text">
                    EduNexus will use your uploaded material
                    to create a structured study schedule.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PLAN SETTINGS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                days = st.number_input(
                    "📅 Study Duration",
                    min_value=1,
                    max_value=60,
                    value=7,
                    step=1
                )

            with col2:

                hours = st.number_input(
                    "⏱️ Hours Per Day",
                    min_value=1,
                    max_value=12,
                    value=3,
                    step=1
                )

            with col3:

                difficulty = st.selectbox(
                    "⚡ Difficulty",
                    [
                        "Beginner",
                        "Intermediate",
                        "Advanced"
                    ]
                )

            focus = st.text_input(
                "🎯 Focus Topic",
                placeholder="Example: Java Collections, DBMS, Operating Systems..."
            )

            goal = st.selectbox(
                "🏆 Study Goal",
                [
                    "Exam Preparation",
                    "Complete the Syllabus",
                    "Quick Revision",
                    "Deep Understanding",
                    "Interview Preparation"
                ]
            )

            # ------------------------------------------------
            # PLAN SUMMARY
            # ------------------------------------------------

            total_hours = days * hours

            st.markdown(
                "### 📊 Plan Overview"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-number">
                        {days}
                        </div>

                        <div class="metric-label">
                        Study Days
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-number">
                        {hours}h
                        </div>

                        <div class="metric-label">
                        Daily Study
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-number">
                        {total_hours}h
                        </div>

                        <div class="metric-label">
                        Total Hours
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c4:

                difficulty_icon = {
                    "Beginner": "🟢",
                    "Intermediate": "🟡",
                    "Advanced": "🔴"
                }

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-number">
                        {difficulty_icon[difficulty]}
                        </div>

                        <div class="metric-label">
                        {difficulty}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------------
            # GENERATE PLAN
            # ------------------------------------------------

            if st.button(
                "✨ Generate My AI Study Plan",
                type="primary",
                use_container_width=True
            ):

                search_topic = (
                    focus.strip()
                    if focus.strip()
                    else "important topics"
                )

                sources = retrieve_information(
                    search_topic,
                    st.session_state.selected_document,
                    number_of_results=10
                )

                if not sources:

                    st.error(
                        "❌ I couldn't find relevant topics "
                        "in your uploaded material."
                    )

                else:

                    context = create_context(
                        sources,
                        maximum_characters=12000
                    )

                    with st.spinner(
                        "🤖 EduNexus is creating your personalized plan..."
                    ):

                        answer = generate_ai_response(
                            f"""
                            Create a detailed {days}-day study plan.

                            Study goal:
                            {goal}

                            Study time:
                            {hours} hours per day.

                            Difficulty:
                            {difficulty}

                            Focus topic:
                            {search_topic}

                            IMPORTANT RULES:

                            1. Use ONLY topics that appear
                               in the supplied study material.

                            2. Do not invent topics.

                            3. Divide every day into:
                               • Learning
                               • Revision
                               • Practice

                            4. Include a final revision
                               before the last day.

                            5. Keep the workload realistic
                               for {hours} hours per day.

                            6. Clearly organize the answer
                               day by day.

                            Use this format:

                            DAY 1
                            Learning:
                            - Topic
                            - Topic

                            Revision:
                            - Topic

                            Practice:
                            - Task

                            Estimated Time:
                            X hours

                            Priority:
                            High/Medium/Low

                            Repeat this structure
                            for every day.

                            Make the plan practical
                            for a college student.
                            """,
                            context
                        )

                    # ----------------------------------------
                    # SAVE GENERATED PLAN
                    # ----------------------------------------

                    st.session_state.study_plan = answer
                    st.session_state.study_plan_topic = search_topic
                    st.session_state.study_plan_days = days
                    st.session_state.study_plan_hours = hours

                    save_activity(
                        "Study Planner",
                        search_topic,
                        st.session_state.selected_document
                    )

            # ------------------------------------------------
            # DISPLAY PLAN
            # ------------------------------------------------

            if "study_plan" in st.session_state:

                st.markdown(
                    "## 🚀 Your Personalized Study Plan"
                )

                st.markdown(
                    f"""
                    <div class="card">

                        <div class="card-title">
                        🎯 Goal: {goal}
                        </div>

                        <div class="card-text">

                        📚 Material:
                        {st.session_state.selected_document}

                        &nbsp;&nbsp; • &nbsp;&nbsp;

                        📅 Duration:
                        {st.session_state.study_plan_days} days

                        &nbsp;&nbsp; • &nbsp;&nbsp;

                        ⏱️ Daily:
                        {st.session_state.study_plan_hours} hours

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                if "completed_tasks" not in st.session_state:

                    st.session_state.completed_tasks = set()

                # Count checkbox states
                completed = len(
                    st.session_state.completed_tasks
                )

                # Use a visual progress estimate
                # based on generated plan
                progress_value = min(
                    completed / max(days * 3, 1),
                    1.0
                )

                st.markdown(
                    "### 📈 Your Progress"
                )

                progress_col1, progress_col2, progress_col3 = st.columns(3)

                with progress_col1:

                    st.metric(
                        "Completed",
                        completed
                    )

                with progress_col2:

                    st.metric(
                        "Study Days",
                        days
                    )

                with progress_col3:

                    st.metric(
                        "Progress",
                        f"{int(progress_value * 100)}%"
                    )

                st.progress(
                    progress_value
                )

                # --------------------------------------------
                # STUDY PLAN CONTENT
                # --------------------------------------------

                st.markdown(
                    "### 📚 Daily Schedule"
                )

                # Split generated response into day sections
                day_sections = re.split(
                    r"(?=DAY\s+\d+)",
                    st.session_state.study_plan,
                    flags=re.IGNORECASE
                )

                displayed_days = 0

                for section in day_sections:

                    section = section.strip()

                    if not section:
                        continue

                    day_match = re.search(
                        r"DAY\s+(\d+)",
                        section,
                        re.IGNORECASE
                    )

                    if not day_match:
                        continue

                    day_number = int(
                        day_match.group(1)
                    )

                    displayed_days += 1

                    # ----------------------------------------
                    # DAY CARD
                    # ----------------------------------------

                    st.markdown(
                        f"""
                        <div style="
                            background:white;
                            border:1px solid #e5e7eb;
                            border-radius:18px;
                            padding:22px;
                            margin:15px 0;
                            box-shadow:
                            0 5px 20px
                            rgba(15,23,42,0.06);
                        ">

                            <div style="
                                display:flex;
                                justify-content:space-between;
                                align-items:center;
                            ">

                                <div style="
                                    font-size:20px;
                                    font-weight:800;
                                    color:#111827;
                                ">
                                    📅 DAY {day_number}
                                </div>

                                <div style="
                                    background:#eff6ff;
                                    color:#2563eb;
                                    padding:7px 12px;
                                    border-radius:20px;
                                    font-size:12px;
                                    font-weight:700;
                                ">
                                    ⏱ {hours} HOURS
                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ----------------------------------------
                    # TASKS
                    # ----------------------------------------

                    lines = section.splitlines()

                    task_number = 0

                    for line in lines:

                        line = line.strip()

                        if not line:
                            continue

                        # Remove markdown bullets
                        clean_line = re.sub(
                            r"^[-•*]\s*",
                            "",
                            line
                        )

                        # Skip headings
                        if re.match(
                            r"DAY\s+\d+",
                            clean_line,
                            re.IGNORECASE
                        ):
                            continue

                        if clean_line.lower() in [
                            "learning:",
                            "revision:",
                            "practice:",
                            "estimated time:",
                            "priority:"
                        ]:
                            st.markdown(
                                f"**{clean_line}**"
                            )
                            continue

                        # Display topic/task as checkbox
                        if line.startswith(
                            ("-", "•", "*")
                        ):

                            task_number += 1

                            task_id = (
                                f"day_{day_number}_"
                                f"task_{task_number}"
                            )

                            checked = st.checkbox(
                                clean_line,
                                key=task_id
                            )

                            if checked:

                                st.session_state.completed_tasks.add(
                                    task_id
                                )

                            else:

                                st.session_state.completed_tasks.discard(
                                    task_id
                                )

                # --------------------------------------------
                # STUDY MOTIVATION
                # --------------------------------------------

                if displayed_days > 0:

                    st.markdown(
                        """
                        <div style="
                            margin-top:25px;
                            padding:22px;
                            border-radius:18px;
                            background:
                            linear-gradient(
                                135deg,
                                #111827,
                                #243b68
                            );
                            color:white;
                            text-align:center;
                        ">

                            <div style="
                                font-size:25px;
                                font-weight:800;
                            ">
                                🔥 Keep Going!
                            </div>

                            <div style="
                                margin-top:8px;
                                color:#dbeafe;
                            ">
                                Small progress every day
                                creates big results.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --------------------------------------------
                # SOURCE MATERIAL
                # --------------------------------------------

                st.markdown(
                    "### 📚 Material Used"
                )

                planner_sources = retrieve_information(
                    st.session_state.study_plan_topic,
                    st.session_state.selected_document,
                    number_of_results=6
                )

                display_sources(
                    planner_sources
                )

# ============================================================
# ANALYTICS
# ============================================================

elif menu == "📊 Learning Analytics":

    st.markdown(
        """
        <div class="hero">

        <h1>
        Learning Analytics 📊
        </h1>

        <p>
        Track how you use your EduNexus workspace.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    activity = get_activity()

    if activity.empty:

        st.info(
            "Start using EduNexus to see your learning activity."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Activities",
                len(activity)
            )

        with col2:
            st.metric(
                "Tutor Questions",
                len(
                    activity[
                        activity["activity_type"]
                        == "AI Tutor"
                    ]
                )
            )

        with col3:
            st.metric(
                "Smart Notes",
                len(
                    activity[
                        activity["activity_type"]
                        == "Smart Notes"
                    ]
                )
            )

        with col4:
            st.metric(
                "Quizzes",
                len(
                    activity[
                        activity["activity_type"]
                        == "Practice Quiz"
                    ]
                )
            )

        chart_data = (
            activity[
                "activity_type"
            ]
            .value_counts()
            .reset_index()
        )

        chart_data.columns = [
            "Activity",
            "Count"
        ]

        figure = px.bar(
            chart_data,
            x="Activity",
            y="Count",
            title="EduNexus Usage"
        )

        st.plotly_chart(
            figure,
            use_container_width=True
        )

        st.dataframe(
            activity,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>

    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:12px;
    ">

    EduNexus • AI-Powered RAG Learning Assistant
    <br>
    Every answer is grounded in your uploaded study material.

    </div>
    """,
    unsafe_allow_html=True
)