#!/bin/bash
# Start MCP server in background
python3 configuration_copilot/mcp_server.py &
# Start Streamlit in foreground
exec streamlit run configuration_copilot/configuration_copilot.py --server.port=8502 --server.enableCORS=false
