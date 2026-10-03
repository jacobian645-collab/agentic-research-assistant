# Acme AI Platform Standards (v3.2)

*Owner: Enterprise Architecture · Classification: Internal · Fictional sample document for the demo knowledge base.*

## Approved model tiers

All production generative-AI workloads must run on Amazon Bedrock unless an exception is approved by the AI Architecture Review Board (AARB).
Model usage is tiered: a reasoning tier (Claude Sonnet class) for planning, synthesis and critique, a fast tier (Claude Haiku class) for high-volume tool-calling loops, and a judge tier (Claude Opus class or a different model family) reserved for evaluation.
Teams must not hard-code model identifiers in application code; identifiers are configuration.

## Data residency

Workloads processing personal data of APAC customers must keep inference inside the APAC geography by using `apac.` cross-region inference profiles or an in-region model. The `global.` inference profiles may only be used for workloads classified Public or Internal with no personal data.
Where an approved model has no `apac.` profile, the workload must fall back to a model tier that does, and record the deviation in the solution design.

## Tool integration standard

All agent tools must be exposed through the Model Context Protocol (MCP). Each MCP server runs as its own deployable unit with its own IAM role, network policy and resource limits. Agents receive only the tools they need (least privilege per agent).
Tools that change external state (write, send, purchase, delete) require human approval before execution.

## Production readiness gates for agentic systems

Before production, an agentic system must pass an offline evaluation on a versioned dataset of at least 50 questions with: mean LLM-as-judge score of at least 4.0 out of 5, mean groundedness of at least 4.2 out of 5, and a citation validity rate of at least 95 percent.
The judge must be calibrated against human labels (see the RAG Evaluation Playbook) and re-run on every prompt or model change.
Every agent run must enforce a hard step limit and a per-request cost cap of USD 0.50.
