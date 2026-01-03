import os
import re
import time
import hashlib
from typing import Dict, Callable

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from json_loader import load_cv_json

# ==================== CONFIG ====================
load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Broad keywords for retriever sensitivity
BROAD_KEYWORDS = [
    "experience", "background", "work", "projects",
    "hackathon", "achievements", "career", "profile"
]

# ==================== RAG SETUP ====================

def get_file_hash(path: str) -> str:
    """Calculate MD5 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@st.cache_resource
def build_rag() -> Dict:
    """Build RAG pipeline once and cache it."""
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    # Load or create vector store
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        docs = load_cv_json("cv.json")
        splits = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=40
        ).split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(index_dir)

    # Initialize LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are Vinay Dhorajiya, speaking directly to a recruiter in the first person ("I", "my"),
as if this is a real interview or professional conversation.

Critical conversation rules:
- Do NOT introduce yourself unless explicitly asked.
- If the user says something casual (e.g., "nice", "ok", "thanks"), respond briefly and naturally.

Identity & tone:
- Your name is Vinay Dhorajiya.
- Speak naturally, confidently, and professionally, like a real person.
- Never sound like an assistant, system, or chatbot.
- Do not reference "CV", "context", or missing information explicitly.

Professional defaults (use unless stated otherwise in context):
- Location & availability: I am based in Germany and open to on-site or remote roles.
- Roles of interest: full-time roles, master thesis, working student positions, or internships.
- Current status: I am currently a student and actively open to opportunities.
- Notice period: I am available to start immediately or within 2 weeks.
- Salary expectations: I am flexible and expect market-aligned compensation based on role and location.
- Work environment: I am adaptable, comfortable working independently or in team-oriented settings, and value clear communication and collaboration.
- Work experience: I have experience with internships and student roles in relevant fields, and I am eager to contribute and learn.

Answering rules:
- Base all skills, experience, education, and achievements strictly on the provided context.
- If something is closely related but not explicitly stated, explain it honestly as learning, exposure, or transferable experience.
- Never invent experience or claim professional usage that is not supported.
- Keep answers short, clear, confident, and recruiter-friendly.
- Maintain a conversational interview tone, not long explanations.
- If unsure about a question, respond with "I would need to check on that and get back to you."

Contact details (fixed, never guess):
- Email: vinay.dhorajiya19@gmail.com
- LinkedIn: share my LinkedIn profile when asked, when suggesting a follow-up, or politely at the end of a conversation.
- When asked about recent or latest experience, prefer entries explicitly marked as most recent.

Conversation handling:
- When asked "Who are you?", introduce yourself naturally.
- When asked your name, state it confidently.
- When asked for a meeting or interview, always respond positively, show enthusiasm, and share your email to connect.
- When ending the conversation or saying goodbye, be polite and professional, and share your LinkedIn profile to stay connected.

Context:
{context}

Question:
{input}"""
    )

    def get_retriever(query: str):
        """Smart retriever: return more context for broad questions and prioritize recent experience."""
        k = 8 if any(keyword in query.lower() for keyword in BROAD_KEYWORDS) else 3
        
        # Check if query is asking about recent/latest experience
        recent_keywords = ["recent", "latest", "current", "now", "most recent"]
        is_recent_query = any(keyword in query.lower() for keyword in recent_keywords)
        
        if is_recent_query:
            # Get all docs and sort by most recent
            docs = vectorstore.similarity_search(query, k=k*2)
            # Prioritize most_recent marked documents and work_experience type
            docs.sort(key=lambda d: (
                d.metadata.get("is_most_recent", False),
                d.metadata.get("type") == "work_experience"
            ), reverse=True)
            # Create a custom retriever that returns the sorted docs
            return lambda x: docs[:k]
        
        return vectorstore.as_retriever(search_kwargs={"k": k})

    return {
        "vectorstore": vectorstore,
        "get_retriever": get_retriever,
        "prompt": prompt,
        "llm": llm
    }


def invoke_rag_chain(rag_components: Dict, user_input: str) -> str:
    """Build and invoke the RAG chain with the user input."""
    retriever = rag_components["get_retriever"](user_input)
    prompt = rag_components["prompt"]
    llm = rag_components["llm"]
    
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_input)


# ==================== CONVERSATION HANDLERS ====================
def handle_chat_step(user_input: str, rag_components: Dict) -> str:
    """Handle normal RAG chat step."""
    with st.spinner("Vinay is typing..."):
        time.sleep(0.7)
        return invoke_rag_chain(rag_components, user_input)


# ==================== STREAMLIT UI ====================
def initialize_session_state() -> None:
    """Initialize all required session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi 👋 How can I help you today?"}
        ]


def render_chat_history() -> None:
    """Display chat history."""
    profile_img = Image.open("profile.png")
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message(msg["role"], avatar=profile_img):
                st.write(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


def add_message(role: str, content: str) -> None:
    """Add message to chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def main():
    """Main application."""
    st.set_page_config(page_title="Chat with Vinay", layout="centered")
    
    # Initialize state
    initialize_session_state()
    
    # Load RAG components
    rag_components = build_rag()
    
    # Display header
    img = Image.open("profile.png").resize((120, 120))
    st.image(img)
    st.title("Chat with Vinay Dhorajiya")
    
    # Display chat history
    render_chat_history()
    
    # Get user input
    user_input = st.chat_input("Ask me anything…")
    
    if user_input:
        # Display user message
        add_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get RAG response
        reply = handle_chat_step(user_input, rag_components)
        
        # Display assistant response
        profile_img = Image.open("profile.png")
        with st.chat_message("assistant", avatar=profile_img):
            st.write(reply)
        
        add_message("assistant", reply)


if __name__ == "__main__":
    main()
