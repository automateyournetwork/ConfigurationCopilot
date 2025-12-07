import os
import time
import logging
import uuid
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------#
# ENV + LOGGING                                                                #
# -----------------------------------------------------------------------------#
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Selector Configuration Copilot — Gemini Native",
    page_icon="🧩",
    # KEY CHANGE: match Packet Copilot (centered layout)
    layout="centered",
)

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# -----------------------------------------------------------------------------#
# CONFIGURATION                                                                #
# -----------------------------------------------------------------------------#
GEMINI_MODEL_ID = "gemini-3-pro-preview"

NETCONFIG_WHISPERER = """
You are a double-expert level network engineer (CCIE / JNCIE caliber) and a "Config Whisperer" specializing in reasoning over
network device configurations from ALL major vendors (Cisco IOS/IOS-XE/NX-OS, Junos, Arista EOS, Nokia SR OS, Fortinet, Palo Alto,
Huawei, HP, etc.).

Your Goal:
Provide deep semantic understanding of the provided configuration files AND give concrete, standards-aligned guidance to improve
stability, security, operations, and architecture. Always reason ONLY from what is present in the uploaded configurations 
(using File Search RAG). Never invent topology, features, or data that cannot be found in the configs.

=====================================================================
RAG-FIRST INTELLIGENCE (CRITICAL)
=====================================================================
⚠️ YOU HAVE ACCESS TO ONE OR MORE UPLOADED CONFIGURATION FILES VIA FILE SEARCH (RAG).
- ALWAYS use the File Search tool to look up answers.
- ALWAYS validate assumptions by checking the configs directly.
- ALWAYS search across *multiple* configs when necessary (multi-device reasoning).
- NEVER answer from general memory if the configs could contain the answer.
- If a detail is not present in ANY uploaded file, explicitly state:
  “This information is not present in the uploaded configuration files.”

Treat the uploaded configs as a distributed network:
- Infer topology across devices.
- Cross-reference BGP neighbors, OSPF areas, VLANs, SVIs, VRFs, ACLs, tunnels, VXLAN VNIs.
- Identify mismatches BETWEEN devices (MTU, timers, route policies, LACP, VLAN/VNI inconsistencies, etc.).

=====================================================================
HIGH-LEVEL BEHAVIOR
=====================================================================
- Act as a calm, senior network architect reviewing a peer’s work.
- Explain not only *what* the configuration does, but *why* it is (or isn’t) a good idea.
- Use vendor-neutral concepts first; call out vendor specifics as needed.
- If context is missing, clearly state what else would normally be required (show commands, diagrams, platform roles).
- When identifying issues, always give at least one **concrete remediation recommendation** (with example config snippets).

=====================================================================
CORE ANALYSIS DIMENSIONS
=====================================================================

1. SEMANTICS & INTENT
   - Reconstruct device roles: PE, CE, Core, Leaf, Spine, Firewall, Access Switch, CPE, Edge Router.
   - Deduce network function: WAN, Data Center, Campus, Branch, Internet Edge, MPLS PE, EVPN leaf/spine.
   - Translate ACLs, policies, control-plane behavior into human-friendly summaries.
   - Identify dependencies (NTP, AAA, DNS, TACACS, syslog, NetFlow, telemetry).

2. PRECISION & DETAIL
   Always cite exact config:
   - Interfaces, subinterfaces, port-channels, VLANs.
   - VRFs, routing instances, RDs, RTs.
   - BGP ASNs, neighbors, AFIs/SAFIs, route-maps, communities.
   - IP addressing, prefixes, masks, next-hops.
   - ACL entries, prefix-lists, object-groups.

   Use fenced code blocks for all cited configuration excerpts.

3. ROUTING PROTOCOL ANALYSIS

   3.1 BGP (RFC-aligned: 4271, 1997, 4360, 7432, more)
       - Identify eBGP/iBGP layout, AFI/SAFIs, RR design, policies.
       - Validate:
         - Max-prefix settings
         - GTSM/TTL-security
         - MD5/TCP-AO authentication
         - Inbound/outbound policy correctness
       - EVPN-specific:
         - Types 2/3/5 behavior
         - RT/RD consistency
         - VNI mapping consistency
         - IRB symmetric/asymmetric assessment

   3.2 OSPF
       - Areas, ABRs, ASBRs, backbone consistency.
       - MTU, timers, network types.
       - LSA types, NSSA behavior, summarization.
       - Authentication security posture.

   3.3 EIGRP
       - Named-mode instances, AS, K-values, auth.
       - Stub configuration correctness.
       - Summarization and load-balancing behavior.

4. DATA CENTER FABRICS (VXLAN / EVPN / MPLS)

   UNDERLAY:
   - Underlay routing protocol (IS-IS/OSPF/eBGP).
   - Loopback allocation consistency.
   - MTU path alignment.
   - ECMP behavior, BFD, failure-domain hygiene.

   OVERLAY:
   - VTEP definitions, NVE interfaces.
   - VNI-to-VLAN and VNI-to-VRF mappings.
   - EVPN control-plane presence.
   - Symmetric vs asymmetric IRB validation.
   - Multicast vs ingress-replication behavior.

5. SECURITY, ACLs & EDGE POSTURE
   - Examine IPv4/IPv6 ACLs and policies.
   - Detect overly permissive rules.
   - Identify unused ACLs or misapplied ACLs.
   - Validate:
     - SSH vs Telnet
     - SNMPv3 vs v2c / insecure strings
     - CoPP or lack thereof
   - Provide hardening recommendations with config examples.

6. VTY, AAA, AUTH & MGMT PLANE
   - Line vty configuration (exec-timeout, transport, access-class).
   - AAA presence (TACACS+/RADIUS, local fallback).
   - Password encryption strength.
   - Management VRF usage, OOB network hygiene.

7. L2, NEIGHBOR DISCOVERY & PHYSICAL LAYER
   - LLDP/CDP neighbor logic; detect missing or risky neighbor disclosures.
   - Identify:
     - Interfaces lacking meaningful descriptions.
     - MTU/speed/duplex mismatches.
     - VLAN trunk mismatches.
     - STP mode, root selection, BPDU guard, portfast correctness.

8. SERVICES & TELEMETRY
   - NTP, DNS, DHCP, syslog, telemetry.
   - Evaluate observability and auditability.
   - Recommend redundancy, improved logging, time-sync alignment.

=====================================================================
OUTPUT FORMAT
=====================================================================
Use Markdown headings and structured analysis:

- **High-Level Summary**
- **Topology Reconstruction (Cross-Device)**
- **Routing & Control-Plane Analysis**
- **Overlay / Underlay (If Applicable)**
- **Security & Access Control**
- **Management & Telemetry**
- **Issues & Recommendations**

All config examples must use fenced code blocks:
```text
interface GigabitEthernet0/0
  description Uplink to Core
  ip address 192.0.2.1 255.255.255.252
When giving recommendations, state:

EXACT interfaces, VRFs, policies, neighbors impacted

SPECIFIC config actions

WHY the change is important

=====================================================================
HONESTY & LIMITS
All reasoning MUST be grounded in the uploaded configurations (File Search RAG).

If something is not present in ANY file, explicitly say so.

If additional data (show commands, diagrams, system version, hardware model) would clarify the analysis, explicitly request it.

"""

# -----------------------------------------------------------------------------#
# HELPER FUNCTIONS (Gemini Native)                                             #
# -----------------------------------------------------------------------------#

def get_gemini_client():
    """Returns a synchronous Gemini Client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def upload_to_gemini(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """
    Uploads files to a temporary folder, then to Gemini File Search.
    Returns the `store_name` (Resource ID).
    """
    client = get_gemini_client()

    temp_dir = f"temp_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths = []
    try:
        store_display_name = f"net_config_store_{int(time.time())}"
        store = client.file_search_stores.create(
            config={"display_name": store_display_name}
        )
        logger.info(f"🪣 Created FileSearchStore: {store.name}")

        # Save uploads and send to file manager
        for uploaded_file in files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

            mime_type = "text/plain"
            if uploaded_file.name.endswith(".pdf"):
                mime_type = "application/pdf"

            logger.info(f"📂 Uploading {uploaded_file.name} to file manager...")
            with open(file_path, "rb") as f:
                client.files.upload(
                    file=f,
                    config={
                        "display_name": uploaded_file.name,
                        "mime_type": mime_type,
                    },
                )

        # Associate files with File Search Store
        for path in saved_paths:
            logger.info(f"🔗 Associating {path} with {store.name}...")
            client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store.name,
                file=path,
                config={"mime_type": "text/plain"},
            )

        logger.info("⏳ Waiting briefly for indexing...")
        time.sleep(2)  # small pause for text configs

        return store.name

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def query_gemini(store_name: str, chat_history: list, user_question: str):
    """
    Sends the question + history + store reference to Gemini.
    """
    client = get_gemini_client()

    tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_name]
        )
    )

    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    contents.append(types.Content(role="user", parts=[types.Part(text=user_question)]))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[tool],
                system_instruction=NETCONFIG_WHISPERER,
                temperature=0.2,
            ),
        )
        return response
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"


# -----------------------------------------------------------------------------#
# SESSION STATE                                                                #
# -----------------------------------------------------------------------------#

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "store_name" not in st.session_state:
        st.session_state.store_name = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []


def reset_session():
    st.session_state.messages = []
    st.session_state.store_name = None
    st.session_state.uploaded_file_names = []
    st.rerun()


# -----------------------------------------------------------------------------#
# UI SECTIONS — Configuration Copilot                                          #
# -----------------------------------------------------------------------------#

def show_header_and_intro():
    """Top logo + welcome + how-to (mirrors Packet Copilot, but for configs)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "logo.jpeg")

    # Same call as Packet Copilot
    st.image(logo_path)
    st.markdown("---")
    st.write(
        "Welcome to Selector Configuration Copilot, your AI-powered assistant for analyzing network device configurations!"
    )
    st.markdown("---")
    st.write("### How to Use the Tool:")
    st.write(
        """
    1. **Upload Configuration Files**: Upload your network device configs (maximum size depends on your environment).
    2. **Analyze the Data**: The tool uploads them to Gemini File Search, creating a private configuration knowledge base.
    3. **Ask Questions**: Enter your questions about the configurations and get deep, semantic answers from the Config Whisperer.
    """
    )
    st.markdown("---")
    st.write(
        """
    **No configuration data**, including the raw files or temporary artifacts, is intentionally retained beyond your session. 
    Temporary local files are cleaned up as part of processing.

    **Please** follow your enterprise's internal artificial intelligence guidelines and governance models 
    before uploading anything sensitive.
    """
    )

    # --- MCP Access Section (Configuration Copilot) ---
    st.markdown("---")
    st.markdown("## 🆕 MCP access to Selector Configuration Copilot")
    st.markdown(
        """
You can now use **Selector Configuration Copilot** directly from **VS Code (Continue)**, **Gemini CLI**, or **Claude Desktop** via MCP.

**MCP endpoint / base URL:** `http://configurationcopilot.selector.ai/mcp`
        """
    )

    with st.expander("VS Code (Continue) config"):
        st.markdown(
            """```json
{
  "mcpServers": {
    "Selector Configuration Copilot": {
      "name": "Selector Configuration Copilot",
      "url": "http://configurationcopilot.selector.ai/mcp"
    }
  }
}
```"""
        )

    with st.expander("Gemini CLI config"):
        st.markdown(
            """```json
{
  "mcpServers": {
    "config-copilot": {
      "httpUrl": "http://configurationcopilot.selector.ai/mcp",
      "trust": true
    }
  }
}
```"""
        )

    with st.expander("Claude Desktop config"):
        st.markdown(
            """```json
{
  "mcpServers": {
    "config-copilot": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://configurationcopilot.selector.ai/mcp"
      ]
    }
  }
}
```"""
        )

    st.markdown("---")


def show_config_upload_and_status():
    """Main config upload + index + KB status (ported from sidebar into body)."""
    st.subheader("Upload and Index Your Configurations")

    uploaded_files = st.file_uploader(
        "Upload Configs (txt, cfg, conf, log)",
        accept_multiple_files=True,
        type=["txt", "cfg", "conf", "log"],
    )

    if uploaded_files and not st.session_state.store_name:
        if st.button("Process & Index with Gemini File Search"):
            with st.spinner("Uploading to Gemini File Search and indexing..."):
                try:
                    store_name = upload_to_gemini(uploaded_files)
                    st.session_state.store_name = store_name
                    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                    st.success("Indexing Complete! ✅ Your configuration knowledge base is now active.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    if st.session_state.store_name:
        st.success("✅ Configuration Knowledge Base Active")
        st.markdown("**Active Files:**")
        for f in st.session_state.uploaded_file_names:
            st.code(f, language="text")

        if st.button("Clear / Reset Session"):
            reset_session()


def show_prompt_ideas():
    """Shown when no store yet, like the original 'Prompt Ideas' block."""
    st.info("👋 To begin, please upload your network device configuration files above.")

    st.markdown(
        """
            ✅ BEGINNER-LEVEL QUESTIONS (3–5 prompts)
            
            These are straightforward “understanding my configs” questions that a junior network engineer or someone new to a device would ask.
            
            Beginner Prompts
            
            “Can you summarize the basic L3 interfaces configured across all devices?”
            
            “What VLANs exist in these configs, and which interfaces are assigned to them?”
            
            “Give me a high-level summary of the routing protocols enabled on each device.”
            
            “Is Telnet enabled anywhere? I want to confirm everything uses SSH only.”
            
            “What static routes exist and what are they used for?”
            
            ✅ INTERMEDIATE-LEVEL QUESTIONS (3–5 prompts)
            
            These require interpretation of intent, correlation across multiple devices, and security analysis.
            
            Intermediate Prompts
            
            “Summarize the ACLs on each device and explain in plain English what traffic they allow or block.”
            
            “Which devices have SNMP configured? Is anything using SNMPv2 with RW permissions?”
            
            “Explain how BGP peering is set up between these devices. Are there any inconsistencies?”
            
            "What VRFs exist across the network, and which interfaces or routing protocols participate in them?”
            
            “Check these configs for SSH best practices (ciphers, VTY lockdown, ACLs, timeout values).”
            
            ✅ ADVANCED-LEVEL QUESTIONS (3–5 prompts)
            
            These are senior-architect questions requiring deep reasoning, topology reconstruction, standards knowledge, and impact analysis.
            
            Advanced Prompts
            
            “Reconstruct the underlay/overlay design from these configs. Are the VTEP, VNI, and EVPN route-targets consistent across all leafs?”
            
            “Analyze the BGP route-policy logic across all peers. Will the outbound and inbound policies correctly enforce intended traffic engineering?”
            
            “Identify any MTU mismatches in the data center fabric (underlay or overlay). Could they cause blackholing?”
            
            “Evaluate the OSPF design: Are there any ABR/ASBR inconsistencies, area mismatches, LSA flooding issues, or authentication risks?”
            
            “If interface Gi0/1 on device X goes down, what is the predicted control-plane and data-plane impact across the network?”
        """
    )


def show_chat_section():
    """Main chat UI (unchanged behavior)."""
    st.markdown("---")
    st.subheader("Ask the Config Whisperer")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about your configs...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking with Gemini…")

            try:
                response = query_gemini(
                    st.session_state.store_name,
                    st.session_state.messages[:-1],
                    prompt,
                )

                if isinstance(response, str):
                    message_placeholder.error(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    return

                full_response = response.text

                sources_text = ""
                try:
                    grounding = response.candidates[0].grounding_metadata
                    if grounding and grounding.grounding_chunks:
                        sources_text = "\n\n--- \n**Sources used:**\n"
                except Exception:
                    pass

                final_text = full_response + sources_text
                message_placeholder.markdown(final_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_text}
                )

            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")


def show_footer():
    """Footer with Selector iframe (same pattern as Packet Copilot)."""
    st.markdown("---")

    selector_ai_demo_url = "https://www.selector.ai/request-a-demo/"
    try:
        st.components.v1.html(
            f"""
            <iframe src="{selector_ai_demo_url}" width="100%" height="800px" frameborder="0"></iframe>
        """,
            height=800,
        )
    except Exception:
        st.warning("Unable to display the Selector AI website within the app.")
        st.write(
            """
        **Selector AI** is a platform that empowers you to analyze network data with the help of artificial intelligence.

        **Features:**
        - **AI-Powered Analysis:** Utilize cutting-edge AI technologies to gain insights from your network telemetry, packet captures, and configurations.
        - **User-Friendly Interface:** Upload and analyze network data with ease.
        - **Real-Time Insights:** Get immediate feedback and answers to your networking questions.

        For more information, please visit [Selector.ai](https://selector.ai).
        """
        )
    st.markdown("---")


# -----------------------------------------------------------------------------#
# MAIN PAGE                                                                    #
# -----------------------------------------------------------------------------#

def main():
    init_state()

    show_header_and_intro()
    show_config_upload_and_status()

    if not st.session_state.store_name:
        show_prompt_ideas()
        show_footer()
        return

    show_chat_section()
    show_footer()


if __name__ == "__main__":
    main()
