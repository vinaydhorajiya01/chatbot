import os
import re
import time
import hashlib
from typing import Dict, Callable
from datetime import datetime

import streamlit as st
import pandas as pd
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

EXCEL_FILE = "visitors.xlsx"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Broad keywords for retriever sensitivity
BROAD_KEYWORDS = [
    "experience", "background", "work", "projects",
    "hackathon", "achievements", "career", "profile"
]

# ==================== VALIDATORS ====================
def is_valid_name(name: str) -> bool:
    """Validate name: 2-50 characters, letters and spaces only."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z ]{1,50}", name.strip()))


def is_valid_email(email: str) -> bool:
    """Validate email format."""
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


def is_valid_company(company: str) -> bool:
    """Validate company: min 2 chars with at least one letter."""
    stripped = company.strip()
    return len(stripped) >= 2 and any(c.isalpha() for c in stripped)


# ==================== VISITOR MANAGEMENT ====================
def save_visitor(visitor: Dict[str, str]) -> None:
    """Save visitor info to Excel file."""
    row = {
        "name": visitor["name"],
        "email": visitor["email"],
        "company": visitor["company"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_excel(EXCEL_FILE, index=False)


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
        """Smart retriever: return more context for broad questions."""
        k = 8 if any(keyword in query.lower() for keyword in BROAD_KEYWORDS) else 3
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
def handle_name_step(user_input: str) -> str:
    """Handle name collection step."""
    if not is_valid_name(user_input):
        return "Could you please share your full name? 😊"
    
    st.session_state.visitor["name"] = user_input.strip()
    st.session_state.step = "ask_email"
    return f"Nice to meet you, {st.session_state.visitor['name']} 😊 What's your email address?"


def handle_email_step(user_input: str) -> str:
    """Handle email collection step."""
    if not is_valid_email(user_input):
        return "That doesn't look like a valid email address. Could you please check and try again?"
    
    st.session_state.visitor["email"] = user_input.strip()
    st.session_state.step = "ask_company"
    return "Thanks! Which company are you representing?"


def handle_company_step(user_input: str) -> str:
    """Handle company collection step."""
    if not is_valid_company(user_input):
        return "Could you please share your company name?"
    
    st.session_state.visitor["company"] = user_input.strip()
    save_visitor(st.session_state.visitor)
    st.session_state.step = "chat"
    return "Perfect, thanks! 😊 How can I help you today?"


def handle_chat_step(user_input: str, rag_components: Dict) -> str:
    """Handle normal RAG chat step."""
    with st.spinner("Vinay is typing..."):
        time.sleep(0.7)
        return invoke_rag_chain(rag_components, user_input)


# ==================== STREAMLIT UI ====================
def initialize_session_state() -> None:
    """Initialize all required session state variables."""
    if "visitor" not in st.session_state:
        st.session_state.visitor = {"name": None, "email": None, "company": None}
    
    if "step" not in st.session_state:
        st.session_state.step = "ask_name"
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi 👋 Before we begin, may I know your name?"}
        ]


def render_chat_history() -> None:
    """Display chat history."""
    for msg in st.session_state.messages:
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
        
        # Process based on current step
        step_handlers = {
            "ask_name": handle_name_step,
            "ask_email": handle_email_step,
            "ask_company": handle_company_step,
            "chat": lambda input_: handle_chat_step(input_, rag_components)
        }
        
        reply = step_handlers[st.session_state.step](user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(reply)
        
        add_message("assistant", reply)


if __name__ == "__main__":
    main()
