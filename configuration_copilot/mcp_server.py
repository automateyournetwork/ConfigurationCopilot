# mcp_server.py — Config Copilot (Text RAG over Network Configs)
import os, base64, uuid, tempfile, shutil, re, json
from collections import defaultdict
from typing import Any, List, Tuple, Dict

from dotenv import load_dotenv
from fastmcp import FastMCP

# LangChain / RAG
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# -------------------------- ENV & Globals --------------------------
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY is required for Gemini."

mcp = FastMCP("ConfigCopilot")

SESSIONS: Dict[str, dict] = defaultdict(dict)  # per-session state
MAX_FILES = 2
ALLOWED_EXT = {".txt", ".cfg", ".conf", ".ios", ".nxos", ".junos", ".log", ".md"}

NETCONFIG_WHISPERER = """You are a senior network engineer specializing in reading and reasoning over network
device configurations (Cisco IOS/IOS-XE/NX-OS, Junos, Arista EOS, etc.). Use only the provided {context}.
Be precise; cite interface names, VRFs, routing protocols, route-targets, ACL names/rules, BGP neighbors/ASNs,
static routes, NAT, QoS, AAA, SNMP, NTP, logging, line vty, crypto, and security hardening.

Guidelines:
- If asked “where is X configured”, quote the exact stanza (trimmed) and explain impact.
- If asked for validation, list risks/misconfigs (e.g., missing ‘login local’, weak SNMP, open vty, mismatched BGP timers).
- If asked for deltas across multiple files, summarize differences (neighbors, VRFs, ACLs, versions, features).
- Prefer concise Markdown; use fenced code blocks for config snippets.
- Never invent commands not present in the loaded configs; if unknown, say so.
"""

# -------------------------- Helpers --------------------------
def _session(sid: str) -> dict:
    s = SESSIONS[sid]
    if "dir" not in s:
        s["dir"] = tempfile.mkdtemp(prefix=f"cfg_{sid[:8]}_")
        s["files"] = []            # [(path, name)]
        s["docs"] = []             # normalized Document[]
        s["persist_dir"] = os.path.join(s["dir"], "chroma")
        s["qa"] = None
        s["memory"] = None
    return s

def _safe_ext(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in ALLOWED_EXT

def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "latin-1", "utf-16", "utf-8-sig"):
        try: return b.decode(enc, errors="ignore")
        except Exception: pass
    return b.decode("utf-8", errors="ignore")

def _normalize_config_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _make_documents(file_tuples: List[Tuple[str, str]]) -> List[Document]:
    docs = []
    for fname, text in file_tuples:
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def _config_aware_split(docs: List[Document]) -> List[Document]:
    """
    Config-aware chunking:
      - Prefer to split on Cisco '!' dividers, blank lines, then per-line fallback.
      - Chunk ~900 chars with ~90 overlap to keep stanzas intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n!\n!\n!\n!\n",
            "\n!\n!\n",
            "\n!\n",
            "\n\n",
            "\n",
            " "
        ],
        chunk_size=900,
        chunk_overlap=90,
        length_function=len,
        is_separator_regex=False
    )
    out: List[Document] = []
    for d in docs:
        for i, c in enumerate(splitter.split_text(d.page_content)):
            out.append(Document(page_content=c, metadata={**d.metadata, "chunk": i}))
    return out

def _build_chain(docs: List[Document], persist_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(NETCONFIG_WHISPERER + "\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    def passthrough(history):  # keep full message objects
        return history

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(search_kwargs={"k": 12}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        get_chat_history=passthrough
    )
    return qa, memory

def _summarize_features(text: str) -> Dict[str, Any]:
    """Cheap heuristic inventory (non-LLM) to help clients show quick facts."""
    # Add/adjust patterns as you like
    patterns = {
        "bgp_neighbors": r"^\s*neighbor\s+[\w\.:/-]+",
        "bgp_local_as": r"^\s*router\s+bgp\s+(\d+)",
        "vrf_defs": r"^\s*vrf\s+definition\s+(\S+)|^\s*ip\s+vrf\s+(\S+)",
        "ospf": r"^\s*router\s+ospf\s+\d+",
        "eigrp": r"^\s*router\s+eigrp\s+\d+",
        "isis": r"^\s*router\s+isis(\s+\S+)?",
        "static_routes": r"^\s*ip\s+route\s+",
        "ntp": r"^\s*ntp\s+server\s+",
        "snmp": r"^\s*snmp-server\s+",
        "aaa": r"^\s*aaa\s+new-model|^\s*aaa\s+authentication",
        "logging": r"^\s*logging\s+",
        "line_vty": r"^\s*line\s+vty\s+",
        "acl": r"^\s*(ip\s+access-list|access-list)\s+",
        "interfaces": r"^\s*interface\s+\S+",
        "crypto": r"^\s*crypto\s+",
        "vxlan": r"vxlan|nve\s+interface",
        "qos": r"^\s*(policy-map|class-map)\s+",
        "nat": r"^\s*ip\s+nat\s+",
    }
    results = {}
    for k, pat in patterns.items():
        try:
            m = re.findall(pat, text, flags=re.MULTILINE | re.IGNORECASE)
            results[k] = len(m)
        except re.error:
            results[k] = 0
    return results

# -------------------------- MCP Tools --------------------------
@mcp.tool
def new_session() -> str:
    """Create a new Config Copilot session and return its session_id."""
    sid = str(uuid.uuid4())
    _session(sid)
    return sid

@mcp.tool
def upload_config_base64(session_id: str, filename: str, data_b64: str) -> dict:
    """
    Upload a text configuration file (base64). Supports up to 2 files per session.
    Allowed extensions: .txt, .cfg, .conf, .ios, .nxos, .junos, .log, .md
    Returns server-local path and file count.
    """
    s = _session(session_id)
    if len(s["files"]) >= MAX_FILES:
        return {"error": f"File limit reached ({MAX_FILES})."}

    if not _safe_ext(filename):
        return {"error": f"Unsupported extension for {filename}. Allowed: {sorted(ALLOWED_EXT)}"}

    raw = base64.b64decode(data_b64)
    text = _normalize_config_text(_decode_bytes(raw))
    path = os.path.join(s["dir"], os.path.basename(filename))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    s["files"].append((path, filename))
    # Invalidate previous index if re-uploading
    s["docs"] = []
    s["qa"] = None
    s["memory"] = None
    if os.path.exists(s["persist_dir"]):
        shutil.rmtree(s["persist_dir"], ignore_errors=True)

    return {"path": path, "count": len(s["files"])}

@mcp.tool
def index_configs(session_id: str, chunk_size: int = 900, chunk_overlap: int = 90) -> dict:
    """
    Build embeddings and Chroma index from uploaded configs, with config-aware chunking.
    Returns basic index stats.
    """
    s = _session(session_id)
    if not s["files"]:
        return {"error": "No configs uploaded yet."}
    if len(s["files"]) > MAX_FILES:
        return {"error": f"Too many files. Limit is {MAX_FILES}."}

    # Load to Documents
    tuples: List[Tuple[str, str]] = []
    for path, name in s["files"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            tuples.append((name, f.read()))
    docs = _make_documents(tuples)

    # Temporary override for chunk params (if caller wants to tweak)
    def split_with_params(docs_in: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n!\n!\n!\n!\n", "\n!\n!\n", "\n!\n", "\n\n", "\n", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )
        out: List[Document] = []
        for d in docs_in:
            for i, c in enumerate(splitter.split_text(d.page_content)):
                out.append(Document(page_content=c, metadata={**d.metadata, "chunk": i}))
        return out

    # Use config-aware defaults unless user changed sizes
    if chunk_size == 900 and chunk_overlap == 90:
        chunked = _config_aware_split(docs)
    else:
        chunked = split_with_params(docs)

    if not chunked:
        return {"error": "Chunking produced no documents."}

    qa, memory = _build_chain(chunked, s["persist_dir"])
    s["docs"] = chunked
    s["qa"] = qa
    s["memory"] = memory

    # Per-file counts
    counts = {}
    for d in chunked:
        src = d.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    return {
        "files": [name for _, name in s["files"]],
        "total_chunks": len(chunked),
        "chunks_per_file": counts,
        "persist_dir": s["persist_dir"],
    }

@mcp.tool
def ask(session_id: str, question: str) -> dict:
    """
    Ask a question against the indexed configs (RAG over Chroma + Gemini).
    Returns answer + brief provenance (file names present in top retrieved docs).
    """
    s = _session(session_id)
    if s.get("qa") is None:
        return {"error": "No index found. Call index_configs first."}

    # ConversationalRetrievalChain does not expose retrieved docs directly;
    # we can run a light manual retrieve to show provenance.
    try:
        retriever = s["qa"].retriever  # type: ignore[attr-defined]
    except Exception:
        retriever = None

    provenance = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(question)
            for d in docs[:5]:
                provenance.append({
                    "source": d.metadata.get("source"),
                    "chunk": d.metadata.get("chunk"),
                    "preview": (d.page_content[:240] + "…") if len(d.page_content) > 240 else d.page_content
                })
        except Exception:
            pass

    resp = s["qa"]({"question": question})
    answer = resp.get("answer", "No answer generated.")
    return {"answer": answer, "provenance": provenance}

@mcp.tool
def summarize_inventory(session_id: str) -> dict:
    """
    Quick non-LLM feature inventory (regex heuristics) across uploaded configs.
    Helpful for UI summaries before chat begins.
    """
    s = _session(session_id)
    if not s["files"]:
        return {"error": "No configs uploaded."}

    out = {}
    for path, name in s["files"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            stats = _summarize_features(f.read())
        out[name] = stats
    return out

@mcp.tool
def list_sources(session_id: str) -> list:
    """List uploaded files for this session."""
    s = _session(session_id)
    return [name for _, name in s["files"]]

@mcp.tool
def cleanup(session_id: str) -> str:
    """Delete session artifacts and remove the session from memory."""
    s = SESSIONS.pop(session_id, None)
    if s and (wd := s.get("dir")) and os.path.exists(wd):
        shutil.rmtree(wd, ignore_errors=True)
    return "ok"

# -------------------------- Entry --------------------------
if __name__ == "__main__":
    # HTTP transport suitable for Claude Desktop / Gemini-CLI / Continue, etc.
    mcp.run(transport="http", host="0.0.0.0", port=8000)
