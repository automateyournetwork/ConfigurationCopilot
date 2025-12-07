# mcp_server.py — Config Copilot (Gemini Native RAG)
import os
import base64
import uuid
import tempfile
import shutil
import logging
import re
import time
from typing import Dict, Any, List
from collections import defaultdict

from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai
from google.genai import types

# -------------------------- ENV & SETUP --------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required.")

# Initialize FastMCP
mcp = FastMCP("ConfigCopilot_Gemini")

# Initialize Gemini Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# SESSION STATE
# Maps session_id -> { "store_id": str, "file_names": list }
SESSIONS: Dict[str, dict] = defaultdict(dict)

# CONSTANTS
GEMINI_MODEL = "gemini-3-pro-preview"
ALLOWED_EXT = {".txt", ".cfg", ".conf", ".ios", ".nxos", ".junos", ".log", ".md"}

# Phase 1 Goal: "Config Whisperer"
NETCONFIG_WHISPERER = """
You are a senior network engineer and "Config Whisperer" specializing in reading and reasoning over network device configurations.

Guidelines:
1.  **Semantics & Intent:** Explain *what* the configuration achieves, not just the commands.
2.  **Precision:** Cite specific interface names, VRFs, BGP ASNs, and IP addresses.
3.  **Validation:** actively look for risks (open SNMP, weak SSH) and mismatches.
4.  **Format:** Use clear Markdown. Use Code Blocks for config snippets.
5.  **Tool Use:** You have access to a File Search tool containing the user's uploaded configs. Use it to answer questions.
"""

# -------------------------- HELPERS --------------------------

def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "store_id": None, 
            "file_names": [],
            "temp_dir": tempfile.mkdtemp() # Keep a temp dir for transient file ops
        }
    return SESSIONS[session_id]

def _safe_ext(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in ALLOWED_EXT

def _summarize_features_regex(text: str) -> Dict[str, int]:
    """
    Phase 0 Capability: Non-LLM Regex correlation.
    Useful for quick inventory without burning tokens.
    """
    patterns = {
        "bgp_neighbors": r"^\s*neighbor\s+[\w\.:/-]+",
        "vrf_defs": r"^\s*vrf\s+definition\s+(\S+)|^\s*ip\s+vrf\s+(\S+)",
        "ospf": r"^\s*router\s+ospf\s+\d+",
        "static_routes": r"^\s*ip\s+route\s+",
        "acls": r"^\s*(ip\s+access-list|access-list)\s+",
        "interfaces": r"^\s*interface\s+\S+",
        "crypto_maps": r"^\s*crypto\s+map\s+",
    }
    results = {}
    for k, pat in patterns.items():
        results[k] = len(re.findall(pat, text, flags=re.MULTILINE | re.IGNORECASE))
    return results

# -------------------------- MCP TOOLS --------------------------

@mcp.tool
def new_session() -> str:
    """Start a new clean session. Returns a session_id."""
    sid = str(uuid.uuid4())
    _get_session(sid)
    return sid

@mcp.tool
def upload_config_base64(session_id: str, filename: str, data_b64: str) -> str:
    """
    Upload a network config file (Base64 encoded) to Gemini File Search.
    Supported: .txt, .cfg, .conf, .ios, .nxos, .junos
    """
    session = _get_session(session_id)
    
    if not _safe_ext(filename):
        return f"Error: Unsupported extension {filename}"

    # 1. Initialize Store if not exists
    if not session["store_id"]:
        try:
            store = client.file_search_stores.create(
                config={"display_name": f"mcp_session_{session_id[:8]}"}
            )
            session["store_id"] = store.name
            print(f"Created Store: {store.name}")
        except Exception as e:
            return f"Error creating store: {e}"

    # 2. Save to temp disk (SDK requires file path)
    file_path = os.path.join(session["temp_dir"], filename)
    try:
        raw_data = base64.b64decode(data_b64)
        with open(file_path, "wb") as f:
            f.write(raw_data)
            
        # 3. Upload to Google
        # We explicitly upload to the store to ensure association
        client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=session["store_id"],
            file=file_path,
            config={"mime_type": "text/plain"} # Force text/plain for configs
        )
        
        session["file_names"].append(filename)
        
        # Wait briefly for indexing (naive polling)
        time.sleep(2) 
        
        return f"Successfully uploaded {filename} to Knowledge Base."

    except Exception as e:
        return f"Upload failed: {e}"

@mcp.tool
def query_configs(session_id: str, question: str) -> str:
    """
    Ask a question about the uploaded network configurations.
    Uses Gemini 3 Pro Preview with File Search.
    """
    session = _get_session(session_id)
    store_id = session.get("store_id")

    if not store_id:
        return "No configurations uploaded yet. Please upload files first."

    # Define the tool connection
    tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_id]
        )
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[tool],
                system_instruction=NETCONFIG_WHISPERER,
                temperature=0.2
            )
        )
        
        # Check for grounding (citations)
        answer = response.text
        if response.candidates[0].grounding_metadata.grounding_chunks:
            answer += "\n\n(Verified against uploaded config files)"
            
        return answer

    except Exception as e:
        return f"Error processing query: {e}"

@mcp.tool
def get_inventory_summary(session_id: str) -> dict:
    """
    Phase 0: Returns a quick Regex count of features (BGP, ACLs, etc) 
    for all uploaded files. DOES NOT use LLM (Fast & Cheap).
    """
    session = _get_session(session_id)
    summary = {}
    
    # We read from the local temp dir we kept
    for fname in session["file_names"]:
        local_path = os.path.join(session["temp_dir"], fname)
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                summary[fname] = _summarize_features_regex(f.read())
                
    return summary

@mcp.tool
def cleanup_session(session_id: str) -> str:
    """Deletes the Google File Search Store and local temp files."""
    session = SESSIONS.get(session_id)
    if not session:
        return "Session not found."

    # Delete Google Store
    if session["store_id"]:
        try:
            client.file_search_stores.delete(name=session["store_id"])
            msg = f"Deleted store {session['store_id']}"
        except Exception as e:
            msg = f"Error deleting store: {e}"
    else:
        msg = "No store to delete"

    # Delete local temp
    if os.path.exists(session["temp_dir"]):
        shutil.rmtree(session["temp_dir"])

    del SESSIONS[session_id]
    return f"Session cleaned up. {msg}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="MCP transport: http (for VS Code / Gemini-CLI) or stdio (for Claude Desktop)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        # HTTP server for VS Code / Gemini-CLI
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        # Stdio server for Claude Desktop
        mcp.run()
