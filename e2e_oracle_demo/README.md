# Agno + Oracle E2E demo

Run Agno agents, tools, teams and workflows with Oracle storage. This demo is maintained on a separate branch and is not part of the Oracle integration PR.

- **STATIC:** deterministic responses and local 64-dimensional synthetic embeddings; no model-provider calls.
- **LIVE:** configured model responses and, when enabled, OCI `cohere.embed-v4.0` embeddings with 1,536 dimensions. Requests can incur provider charges.
- Missing live configuration causes a clearly labelled static fallback. A configured provider failure is an error, not a successful static run.

The Alice/Bob selector demonstrates data partitioning; it is not authentication. The server binds to loopback and is intended for individual development/testing, not public deployment.

## Prerequisites

Use the Oracle integration code in this branch, Python 3.12, an isolated virtual environment and an approved Oracle database/account. Vector search needs Oracle 23ai or newer; real OCI embedding tests need model access and a valid OCI signing profile.

The environment needs Agno and its Oracle dependencies, FastAPI, Uvicorn, `python-multipart`, `openai`, `oci`, `oci-openai` and `httpx`. Tests additionally use pytest and pytest-asyncio. Use your approved dependency installation process. Never copy an existing credential directory into the repository.

The commands below assume the repository root and a `.venv` environment.

## Private deployment settings

Initialize the table namespace and database target guard. Substitute your approved values; the example hostname is deliberately not a real server:

```bash
.venv/bin/python -m e2e_oracle_demo.cli init \
  --db-host db.example.invalid \
  --db-user DEMO_USER \
  --db-container DEMO_PDB \
  --db-port 1521
```

Repeat `--db-host` to approve additional hostnames. With no host option, only `localhost` and `127.0.0.1` are approved. The application retains exact host/account/port checks and verifies the connected database identity. Initialization refuses to replace an existing deployment; review existing settings privately before changing targets.

| Private file | Purpose |
|---|---|
| `~/.local/state/agno-oracle-demo-v2/deployment.json` | Table prefix and approved database user, container, port and `expected_hosts` |
| `~/.local/state/agno-oracle-demo-v2/ownership.json` | Ownership record protecting the demo's table namespace |
| `~/.config/agno-oracle-demo/oracle-url` | Oracle connection URL |
| `~/.config/agno-oracle-demo/oci-iam.json` | OCI model, region, compartment and signing-profile references |
| `~/.oci/config` and its referenced signing key | OCI authentication |

Store the Oracle URL through the hidden prompt:

```bash
.venv/bin/python -m e2e_oracle_demo.cli configure
```

Do not put the URL, passwords, signing keys or API tokens in shell arguments, source files, screenshots or Git. Private settings are user-owned files with mode `600`. File permissions are not encryption. Preserve deployment/ownership state when restarting so the application continues using its existing tables.

A deployment created before configurable host allowlists were introduced needs its approved hostnames added privately to `deployment.json` under `expected_hosts`. Do not remove the guard or publish actual hostnames to work around validation.

## OCI IAM chat

Configure the approved compartment, model ID and signing profile through:

```bash
.venv/bin/python -m e2e_oracle_demo.cli configure-iam
```

The demo targets `us-chicago-1`. Select an approved chat model, for example `openai.gpt-oss-120b`. The OCI SDK loads the profile/key; a signing private key is not an OpenAI API token. Do not commit compartment/user/tenancy identifiers or the OCI profile.

The saved output-token limit applies per model response, not to input history or the sum of Team/Workflow calls. Configuration readiness alone does not establish successful model access. The bottom UI indicator becomes green only after a completed live response in that page context.

For a separately approved token-based provider, `configure-model` stores `model.json` and `openai-api-key` privately. IAM configuration takes precedence when present.

## Real OCI embeddings

Privately merge this `embedding` field into the existing `oci-iam.json`; preserve the other fields:

```json
{
  "embedding": {
    "enabled": true,
    "provider": "oci",
    "model_id": "cohere.embed-v4.0",
    "region": "us-chicago-1",
    "dimensions": 1536,
    "truncate": "NONE"
  }
}
```

The adapter reuses the parent IAM profile and compartment. Documents use `SEARCH_DOCUMENT`; vector queries use `SEARCH_QUERY`. SDK calls run in worker threads for async methods. Provider errors never substitute local vectors.

The embedding model/dimensions/provider identity selects a separate Oracle vector table. Existing synthetic documents remain in their original tables: upload them again in LIVE mode to generate real vectors. For keyword/hybrid searches, use **Documents → Database index setup** for the new table. Index creation changes the database and can require privileges/resources; inspect actual search results and Oracle index status.

## Start and access

Start the loopback-only service from the repository root:

```bash
.venv/bin/python -m e2e_oracle_demo.cli serve
```

For a remote VM, run this on your laptop, substituting your own SSH account and host:

```bash
ssh -N -o ServerAliveInterval=30 \
  -L 127.0.0.1:7777:127.0.0.1:7777 \
  YOUR_SSH_USER@YOUR_VM_HOST
```

Open <http://127.0.0.1:7777/>. A background service manager may launch the same command with the repository as its working directory. Keep service configuration private and restart it after changing application/provider settings. No paid provider call occurs merely from opening the page or requesting status.

## Manual walkthrough

1. Select STATIC or LIVE, Alice/Bob and Agent/Team/Workflow.
2. Send an arbitrary message; inspect effective mode, fallback and saved-run evidence.
3. In LIVE, ask the model to use a tool, for example to add two numbers. Verify both the call and its saved result.
4. Reload and reopen the session. Test history across an actual application restart separately.
5. In Documents, upload a passage, select Vector search and try a semantic paraphrase. Inspect `synthetic_embeddings`, `embedding_model` and `embedding_dimensions` in the response.
6. Repeat under Bob and verify owner separation. The selector does not authenticate a real person.
7. Use the Validation tab's 20 scenarios for memory, learnings, approvals, teams/workflows, large tool results, concurrency, async operations, metrics and bulk/configuration checks. Scheduling, tracing and evaluations require separate explicit integration/validation.
8. Record only observed results. The checklist is browser-local and separated by effective mode; it does not run tests or mark passes automatically. Do not paste credentials or private conversations into its notes.

Background execution in this demo is in-process, not a durable queue. A completed chat does not prove automatic tracing or model-judged evaluation. Static vector retrieval does not establish semantic relevance. Focused semantic smoke tests do not establish relevance for every corpus.

## Verification

Offline tests use isolated fixtures and deny network access:

```bash
.venv/bin/python -m pytest -q \
  e2e_oracle_demo/test_demo.py e2e_oracle_demo/test_embeddings.py
```

Oracle acceptance requires explicit opt-in and the private deployment configuration:

```bash
AGNO_DEMO_ALLOW_ORACLE_TESTS=1 \
  .venv/bin/python -m pytest -q e2e_oracle_demo/test_oracle_live.py
```

To include configured real-embedding document cases, additionally set `AGNO_DEMO_ALLOW_EMBEDDING_TESTS=1`. These provider calls can incur charges. Do not interpret skipped cases as passes. The ordinary acceptance suite is not a complete live-LLM validation.

```bash
.venv/bin/ruff format --check e2e_oracle_demo
.venv/bin/ruff check e2e_oracle_demo
```

Keep generated reports, environment files and credentials outside Git. The running demo's logs and any local test results are deployment-specific and are not published as acceptance claims in this guide.
