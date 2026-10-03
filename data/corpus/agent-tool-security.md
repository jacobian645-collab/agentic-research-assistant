# Security Standard for Agent Tools

*Owner: Security Architecture · Classification: Internal · Fictional sample document for the demo knowledge base.*

## Threats in scope

Indirect prompt injection through web pages and documents, server-side request forgery (SSRF) through URL-fetching tools, sandbox escape from code execution, data exfiltration through tool arguments, and excessive agency (OWASP LLM06).

## Required controls

- Treat all tool output as untrusted data: delimit it, scan it for injection patterns and never let it change the agent's instructions or tool permissions.
- URL fetchers must block private, loopback and link-local addresses, including cloud metadata endpoints, and re-check the destination after redirects.
- Code execution must run in an isolated sandbox with no network, no credentials, CPU and memory limits and a wall-clock timeout. Multi-tenant workloads require micro-VM isolation such as Firecracker or gVisor.
- Each agent receives only the tools it needs. Retrieval agents cannot browse; web agents cannot execute code.
- Every claim in a generated report must be traceable to a tool output; fabricated citations are treated as a defect.
