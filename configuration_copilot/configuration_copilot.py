# streamlit_app_text_rag.py
import os
import re
import shutil
import uuid
import logging
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# -----------------------------------------------------------------------------
# ENV + LOGGING
# -----------------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # not required here but kept if you later swap embeddings/LLM

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Selector Configuration Copilot — Config RAG",
    page_icon="🧩",
)

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")

# -----------------------------------------------------------------------------
# CONFIG SYSTEM PROMPT
# -----------------------------------------------------------------------------
NETCONFIG_WHISPERER = """You are a senior network engineer specializing in reading and reasoning over network device configurations
(Cisco IOS/IOS-XE/NX-OS, Junos, Arista EOS, etc.). Use only the provided {context} to answer.
Be precise, cite interface names, VRFs, routing protocols, route-targets, ACL names/rules, BGP neighbors/ASNs, static routes, NAT,
QoS policies, AAA, SNMP, NTP, logging, line vty, crypto, and security hardening (where applicable).

Guidelines:
- If asked for “where is X configured”, quote the exact stanza (trimmed) and explain impact.
- If asked for validation, list risks/misconfigs (e.g., missing ‘login local’, weak SNMP, open vty, mismatched BGP timers).
- If asked for deltas, summarize differences across the uploaded configs (neighbors, VRFs, ACLs, versions, features).
- If asked for “best practices”, keep it short and actionable.
- Never invent commands that don’t exist in the loaded configs. If unknown, say so.

Return clear, concise Markdown with code blocks for config snippets.
"""

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    # CPU-friendly, small + good quality
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def new_session_id() -> str:
    return str(uuid.uuid4())

def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = new_session_id()
    return st.session_state["session_id"]

def log_session(question: str, answer: str, uploaded_names: List[str]):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        session_id = ensure_session_id()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"config_rag_{session_id}_{ts}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Session ID: {}\nTimestamp: {}\nFiles: {}\n\nQ: {}\n\nA:\n{}\n".format(
                    session_id, ts, ", ".join(uploaded_names), question, answer
                )
            )
    except Exception as e:
        st.warning(f"Log error: {e}")

def decode_bytes(b: bytes) -> str:
    # Robust decode (configs sometimes have odd encodings)
    for enc in ("utf-8", "latin-1", "utf-16", "utf-8-sig"):
        try:
            return b.decode(enc, errors="ignore")
        except Exception:
            continue
    # last resort
    return b.decode("utf-8", errors="ignore")

def normalize_config_text(text: str) -> str:
    # Light normalization: strip nulls, unify line endings, remove excessive whitespace at line ends
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    # Collapse >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def make_documents(file_tuples: List[Tuple[str, str]]) -> List[Document]:
    """
    file_tuples: List of (filename, text)
    """
    docs = []
    for fname, text in file_tuples:
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def config_aware_split(docs: List[Document]) -> List[Document]:
    """
    Config-aware chunking:
    Prefer to break on '!' banners / blank lines / stanzas; then fall back to lines.
    """
    # Prefer big logical boundaries first, then smaller
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n!\n",          # common Cisco section boundary
            "\n!\n!\n",
            "\n!\n!\n!\n",
            "\n!\n!\n!\n!\n",
            "\n\n",           # blank lines
            "\n",             # line by line fallback
            " "               # word fallback
        ],
        chunk_size=900,       # configs are terse; ~900 chars keeps stanzas intact
        chunk_overlap=90,
        length_function=len,
        is_separator_regex=False
    )
    out: List[Document] = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            out.append(Document(page_content=c, metadata={**d.metadata, "chunk": i}))
    return out

def nuke_chroma(persist_dir: str):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# RAG CLASS
# -----------------------------------------------------------------------------
class ChatWithConfigs:
    def __init__(self, docs: List[Document], system_text: str):
        self.embedding = load_embeddings()
        self.docs = config_aware_split(docs)

        # Per-session Chroma
        session_id = ensure_session_id()
        self.persist_dir = f"chroma_cfg_db_{session_id}"
        nuke_chroma(self.persist_dir)  # fresh index per upload
        self.vstore = Chroma.from_documents(self.docs, self.embedding, persist_directory=self.persist_dir)

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", temperature=0.3)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_text + "\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        def passthrough(history):
            return history

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vstore.as_retriever(search_kwargs={"k": 12}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            get_chat_history=passthrough
        )

    def ask(self, question: str) -> str:
        resp = self.qa({"question": question})
        return resp.get("answer", "No answer generated.")

# -----------------------------------------------------------------------------
# UI (Wizard-style without sidebar: Upload -> Build -> Chat)
# -----------------------------------------------------------------------------
def _init_step_state():
    if "step" not in st.session_state:
        st.session_state["step"] = 1
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "uploaded_names" not in st.session_state:
        st.session_state["uploaded_names"] = []
    if "docs" not in st.session_state:
        st.session_state["docs"] = None
    if "chat_instance" not in st.session_state:
        st.session_state["chat_instance"] = None

def _reset_everything():
    sid = st.session_state.get("session_id")
    if sid:
        nuke_chroma(f"chroma_cfg_db_{sid}")
    for k in ["session_id", "step", "uploaded_files", "uploaded_names", "docs", "chat_instance"]:
        st.session_state.pop(k, None)
    _init_step_state()

def _header():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "logo.jpeg")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=False)
    st.title("Selector Configuration Copilot — Network Config RAG")
    st.write("""
This tool helps you **analyze network device configurations** using Retrieval-Augmented Generation (RAG).  
It supports Cisco IOS/IOS-XE/NX-OS, Junos, Arista EOS, and more.

### 🔑 How it works
1. **Upload Configs** — Choose up to 2 configuration files (formats: `.txt, .cfg, .conf`).  
2. **Automatic Indexing** — Your configs are chunked and embedded into a private, per-session vector store (Chroma).  
   - No manual steps.  
   - Everything is reset when you clear or restart.  
3. **Ask Questions** — Type natural language questions like:
   - *“Where are BGP neighbors defined?”*  
   - *“Which VRFs are configured?”*  
   - *“Are R1 and SW1 set up for router-on-a-stick between VLAN 10 and 20?”*  
4. **Get Answers** — Copilot cites exact config stanzas, explains risks/misconfigs, and highlights deltas across devices.

### 🛡️ Data & Governance
- **Ephemeral**: Configs, vector stores, and chat history are wiped when you reset.  
- **Private**: Nothing is uploaded outside this session.  
- **Governance**: Please follow your enterprise AI guidelines before uploading sensitive configs.

---

➡️ **Next Step**: Upload your configuration files below. We’ll handle the rest automatically.
    """)
    st.markdown("---")

def _step1_upload():
    st.subheader("Step 1 — Upload up to 2 configuration files")
    st.markdown("**Supported:** `.txt, .cfg, .conf`")

    uploaded = st.file_uploader(
        "Choose 1–2 files",
        type=["txt", "cfg", "conf"],
        accept_multiple_files=True,
        key="uploader_configs"
    )

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Clear Selection", key="clear_selection"):
            st.session_state["uploaded_files"] = []
            st.session_state["uploaded_names"] = []
            st.rerun()
    with c2:
        if st.button("Reset Session", key="reset_from_upload"):
            _reset_everything()
            st.rerun()

    # ⬇️ AUTO-BUILD right after a valid upload
    if uploaded:
        if len(uploaded) > 2:
            st.warning("Please upload **no more than 2** files.")
            return

        texts, names = [], []
        for f in uploaded:
            names.append(f.name)
            content = decode_bytes(f.getvalue())
            content = normalize_config_text(content)
            texts.append((f.name, content))

        if not texts:
            st.warning("No readable text found.")
            return

        # if user re-uploads, wipe old index
        sid = st.session_state.get("session_id")
        if sid:
            nuke_chroma(f"chroma_cfg_db_{sid}")

        docs = make_documents(texts)
        st.session_state["uploaded_files"] = texts
        st.session_state["uploaded_names"] = names
        st.session_state["docs"] = docs

        with st.spinner("Indexing configs into Chroma…"):
            st.session_state["chat_instance"] = ChatWithConfigs(
                docs=st.session_state["docs"],
                system_text=NETCONFIG_WHISPERER
            )

        st.success("Vector store ready. Moving to Chat…")
        st.session_state["step"] = 2
        st.rerun()

    if st.session_state["uploaded_names"]:
        st.markdown("**Selected files:**")
        for n in st.session_state["uploaded_names"]:
            st.write(f"- {n}")

def _step3_chat():
    st.subheader("Step 2 — Chat with your configs")
    if not st.session_state.get("chat_instance"):
        st.warning("No index found. Upload configs to begin.")
        return

    # full-width input
    user_q = st.text_input(
        "Ask a question about the configs:",
        placeholder="e.g., Are R1 and SW1 configured for router-on-a-stick between VLAN 10 and 20?",
        key="chat_input"
    )

    # action buttons in columns only (no content rendering inside!)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        back_clicked = st.button("← Back to Upload", key="back_to_upload", use_container_width=True)
    with c2:
        send_clicked = st.button("Send", key="send_q", type="primary", use_container_width=True)
    with c3:
        reset_clicked = st.button("Reset Session", key="reset_in_chat", use_container_width=True)

    if back_clicked:
        st.session_state["step"] = 1
        st.rerun()

    if reset_clicked:
        _reset_everything()
        st.rerun()

    # compute the answer, then render it OUTSIDE the columns
    if send_clicked and user_q.strip():
        with st.spinner("Reasoning…"):
            ans = st.session_state["chat_instance"].ask(user_q.strip())
            log_session(user_q, ans, st.session_state.get("uploaded_names", []))
        st.session_state["last_q"] = user_q.strip()
        st.session_state["last_a"] = ans

    # ----- FULL-WIDTH RENDER -----
    if st.session_state.get("last_a"):
        st.markdown("### 🧠 Answer")
        st.markdown(st.session_state["last_a"])

    st.markdown("### 💬 Chat History")
    for m in st.session_state["chat_instance"].memory.chat_memory.messages:
        role = "You" if m.type == "human" else "Configuration Copilot"
        st.markdown(f"**{role}:**\n\n{m.content}")

def main():
    _init_step_state()
    _header()
    st.markdown("---")

    # 2-step flow: Upload (step==1) -> Chat (step==2)
    if st.session_state["step"] == 1:
        _step1_upload()
    else:
        _step3_chat()

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Start Over", key="start_over_footer"):
            _reset_everything()
            st.rerun()
    with c2:
        st.caption("No configs or vectors are retained across resets. Follow your internal governance before uploading sensitive data.")

    st.markdown("---")  # Adds a horizontal line

    selector_ai_demo_url = "https://www.selector.ai/request-a-demo/"
    try:
        st.components.v1.html(f"""
            <iframe src="{selector_ai_demo_url}" width="100%" height="800px" frameborder="0"></iframe>
        """, height=800)
    except Exception as e:
        st.warning("Unable to display the Selector AI website within the app.")
        st.write("""
        **Selector AI** is a platform that empowers you to analyze network Configuration captures with the help of artificial intelligence.

        **Features:**
        - **AI-Powered Analysis:** Utilize cutting-edge AI technologies to gain insights from your network data.
        - **User-Friendly Interface:** Upload and analyze Configuration captures with ease.
        - **Real-Time Insights:** Get immediate feedback and answers to your networking questions.

        For more information, please visit [Selector.ai](https://selector.ai).
        """)
    st.markdown("---")

if __name__ == "__main__":
    main()