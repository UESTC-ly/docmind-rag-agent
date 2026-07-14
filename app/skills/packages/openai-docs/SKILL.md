---
name: "openai-docs"
description: "Answer OpenAI product and API questions from the host-configured official documentation MCP server."
---

# OpenAI Docs through DocMind MCP

Use only DocMind's `call_mcp` action. The deployment config defines the MCP
server URL, authentication token environment variable, headers, and exact tool
allowlist. Do not run copied helper scripts, browse arbitrary domains, or invent
an unavailable tool name.

Workflow:

1. Call the configured documentation search tool with a compact query.
2. Fetch the most relevant official page with the configured fetch tool when
   that tool is available.
3. For API schemas or required fields, use an allowlisted OpenAPI/spec tool when
   configured.
4. Base the answer only on successful MCP observations, distinguish current
   documentation from inference, and include returned official URLs when
   present.
5. If the required MCP server/tool is not configured or fails, state that the
   authoritative lookup is unavailable; do not silently answer from memory as
   though it were verified.

The package itself grants no network authority. A tool is callable only when it
is declared here, enabled by the host, and present in the host allowlist.
