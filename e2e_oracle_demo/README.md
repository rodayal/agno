# Fresh Agno + Oracle dual-mode demo

**Current Live blocker:** OCI rejects the configured key as expired on 30 June 2026. Oracle/static-embedding browser checks passed; a real LLM response is not yet verified. The service and SSH tunnel are running, with STATIC selected for use until the user privately supplies a valid key.

Run Agno agents, tools, teams and workflows against real Oracle storage in either **static** or **live** mode. STATIC responses and embeddings are synthetic. LIVE responses use the configured LLM; document vectors use the configured OCI embedding model when enabled. A changed embedding identity uses a separate vector table: upload documents again in LIVE mode, then use the Documents panel's index setup for keyword/hybrid search.

**Live OCI chat with static embeddings.** The user has saved a model key and selected `openai.gpt-oss-20b`. Live chat uses that provider; all document embeddings remain local and synthetic. Current test results are recorded in [TEST_LOG.md](TEST_LOG.md). Previous static-mode and large-tool-output results are recorded in [TEST_LOG.md](TEST_LOG.md); they do not validate the new live configuration. The browser's 20 checklists remain your own manual notes, not automatically assigned passes.

## Modes and evidence

Choose static/live in the **browser for each request**; HTTP clients send the `mode` field. Mode is not selected through an environment variable or service configuration.

| Selection/configuration | Effective behavior | What it can establish |
|---|---|---|
| Static selected | A synthetic implementation of Agno `Model` emits responses/tool calls; Agno executes registered tools and persists to Oracle. | Actual Agno orchestration, tool execution and storage, when observed. |
| Live selected, key or required model configuration absent | Automatically effective **static**, with a prominent notice, reason and response provenance. Writes use the static namespace. | The same static integration scope; live correctness remains `NOT RUN`. |
| Live selected, configuration present | Agno `OpenAIResponses` calls the explicitly configured provider/model. | Live behavior only after an explicit successful request and evidence review. |
| Configured provider/key/model/network failure | A visible error; no synthetic replacement. | Failure evidence, not a passing live or static result. |

Opening the page, startup, `/api/status`, `/api/inventory` and CLI `check` perform read-only Oracle readiness/inspection work without paid provider calls. Initial setup may record namespace ownership in local state. Missing live configuration is a labelled mode decision; a configured request failing is a different branch and must remain an error.

Static embeddings use a deterministic **SHA256 bag-of-words** scheme. They are synthetic token features, can collide and do not demonstrate semantic understanding. Both chat modes use this same local 64-dimensional embedding algorithm, without any embedding API. Static/live still have separate storage namespaces and embedding tables. This checks Oracle retrieval plumbing, not semantic embedding quality.

Session IDs are checked against owner and effective mode on the server, including chat reuse, mutations, tool context and approval continuation. Job polling requires the owner and **submitted request mode**, while its result reports the effective mode. A session made while live falls back to static remains a static session after real live configuration is added: create a new live session. Alice/Bob illustrate isolation; selecting a name is **not authentication** and does not prevent someone impersonating that name.

## Files, credentials and preservation

| Item | Location |
|---|---|
| Fresh local source | `/Users/rodayal/Documents/Inte/oracle-e2e-fresh-20261026/e2e_oracle_demo` |
| Intended VM checkout | `/scratch/rodayal/agno-pr-10265` |
| Intended VM package | `/scratch/rodayal/agno-pr-10265/e2e_oracle_demo` |
| New local state on the server | `~/.local/state/agno-oracle-demo-v2/` |
| Persistent Oracle credential | `~/.config/agno-oracle-demo/oracle-url` |
| Model settings | `~/.config/agno-oracle-demo/model.json` |
| Provider key | `~/.config/agno-oracle-demo/openai-api-key` |
| Existing service name | `agno-oracle-demo.service` |
| Loopback listener | `127.0.0.1:7777` |

The existing Oracle URL is reused privately; do not print it, place it in a command argument or return it to the browser. Credential files are user-owned with permission `600` and their configuration directory is private (`700`); permissions are not encryption. Model configuration and API keys stay outside Git. Use the server's interactive configuration flow; this guide contains no database URL or key examples.

Fresh deployment state records an isolated owned table prefix. Inspect only that installation's registered tables. The existing service configuration, old Oracle tables, conversations, vectors and credentials are preserved. This build uses the existing `agno-oracle-demo.service` and its CLI `serve` entry point. It has been restarted and checked; do not start a second listener on port 7777.

**Identity changes:** prior hash/content-identity schemes are not migrated by this demo. Use fresh owned tables and reingest source documents into the appropriate static/live embedding table. Reembedding is required when embedding identity/dimensions change. This is a fresh-demo/reingestion plan, not a production migration or a reason to drop old tables.

## Manual setup and model configuration

The package is installed. These commands let you inspect or resume it independently on the VM. Do not start another listener while the existing service owns port 7777.

```bash
cd /scratch/rodayal/agno-pr-10265
.venv/bin/python -m e2e_oracle_demo.cli init
.venv/bin/python -m e2e_oracle_demo.cli check
.venv/bin/python -m e2e_oracle_demo.cli inventory
```

`init` creates deployment metadata if absent and retains it on subsequent calls; runtime validates/records ownership. The private Oracle URL already exists, so the recommended setup is **init, then check**. Runtime reads that existing file privately. Review the recorded prefix before writes. `check` is keyless readiness, not the mutating `/api/checks` regression runner and not a live acceptance suite. `inventory` provides owned-table proof, not permission to change arbitrary Oracle objects. No reset/drop command is part of the default workflow.

Only if the Oracle credential file is missing, run the interactive server command below. `configure` **refuses an existing credential**; it neither reuses nor overwrites it:

```bash
.venv/bin/python -m e2e_oracle_demo.cli configure
```

Static mode needs Oracle configuration but no provider key. For an intentional foreground run, with the existing listener stopped manually first:

```bash
.venv/bin/python -m e2e_oracle_demo.cli serve
```

For laptop access, run this in a **separate laptop terminal** and leave it running:

```bash
ssh -N -o ServerAliveInterval=30 \
  -L 127.0.0.1:7777:127.0.0.1:7777 \
  rodayal@100.94.147.94
```

If a tunnel already owns local port 7777, use that tunnel. Open [the loopback interface](http://127.0.0.1:7777/) on the laptop and select the requested mode in the browser. Stopping the tunnel does not stop the server.

To configure live access later, run privately in the server terminal:

```bash
cd /scratch/rodayal/agno-pr-10265
.venv/bin/python -m e2e_oracle_demo.cli configure-model
```

Choose an accessible Responses model explicitly; **no default chat model is selected**. Configure the embedding model and its dimensions explicitly, and enter the API key through hidden input. An optional **HTTPS `base_url`** may target an approved Responses-compatible gateway. Compatibility must be verified for the selected chat and embedding endpoints; a gateway URL does not establish support for every provider. Other providers need a provider-factory adapter and their own validation.

Saving configuration makes no paid call and does not verify credentials, model access or gateway compatibility. Restart the service to load new provider settings, then select **Live in the browser** for the requests you intend to run. The CLI refuses to overwrite existing model/key files; review or rotate them privately. A configured provider failure must be reported without static substitution. Live chat and teams/workflows can incur charges; invoke them individually. LIVE document ingestion/vector search calls OCI embeddings when enabled; STATIC makes no embedding API calls. Separately configured provider-backed evaluations can also incur charges, but are not mounted by this application. Do not automatically run a paid batch.

### OCI chat configuration

The existing live factory uses Agno `OpenAIResponses`. At the next authorized service startup, it reads the user's saved `model_id` and `base_url`, and passes the separate raw API key to the client. The model ID is not hardcoded in application logic: the user's selection of `openai.gpt-oss-20b` takes effect through the saved settings. The three `OPENAI_*` environment variables are not this demo's configuration source.

The OCI OpenAI-compatible base URL must be:

```text
https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1
```

`model.json` may optionally contain `max_output_tokens`. If omitted, the application now uses **1024 output tokens per model response**. Valid limits are 1 through 16384. The limit is passed to every live agent/member model; it is not a cap on input history, total conversation tokens or the sum of model calls in a Team/Workflow. An existing explicitly saved limit is retained. Provider-side response storage is disabled, the request timeout is 60 seconds and automatic SDK retries are disabled. Oracle continues to store the Agno session/run history. Model access, parameter compatibility and response quality still need live verification.

LIVE document ingestion, search and the agent's search_documents tool now use OCI EmbedText with `cohere.embed-v4.0` (1536 float dimensions), configured by the `embedding` section in `~/.config/agno-oracle-demo/oci-iam.json`. This uses the same SDK/IAM path as `~/.config/agno-oracle-demo/embed_text_demo.py`; that file is a standalone example, not imported or executed by the web service. Documents use SEARCH_DOCUMENT and queries SEARCH_QUERY. The OCI SDK loads the configured signing profile normally. STATIC mode remains local. Provider errors never substitute static vectors. The new embedding identity selects a separate Oracle vector table, preserving existing static data; upload documents again in LIVE mode. The status page shows configuration readiness, not proof of a successful embedding request. Live embedding requests can incur charges.

The browser shows the configured provider/model and output limit after startup. Its bottom indicator becomes green only after a completed live response; configuration alone does not count as a successful LLM call. Startup reads the saved credentials through the application without displaying them. The user has authorized starting the demo and testing it through an SSH tunnel. There is no automated provider call just to check readiness.

Service operations on the server, with one normal restart after installation and test approval:

```bash
systemctl --user status agno-oracle-demo.service --no-pager
systemctl --user restart agno-oracle-demo.service
```

If the unit is stopped, use `systemctl --user start agno-oracle-demo.service` instead. The deployment retains the existing service name and CLI entry point; `init` and `serve` do not install or restart a unit. Stopping a process preserves committed Oracle data and private configuration, but clears the in-process background job list.

## Browser workflow

The bottom **LLM status indicator** is separate from your manual test verdicts:

- Grey: static selected, live unavailable/static fallback, or availability unknown.
- Amber: live configuration is present but no completed live response has been verified in this page context, or a request is running.
- Green: the last completed response in this page context was reported as effective live, non-fallback and non-simulated. This is response evidence, not a continuous uptime monitor or proof that embeddings are working.
- Red: a live request was incomplete or failed. Inspect the error; it is not a passing live test and is not silently replaced by static output.

Reloading or changing user, mode or component clears the response-verification state. No model request is made just to turn the light green. The live-success path remains unverified against a real provider until you configure your key and send a request.

Choose Alice or Bob, requested mode, and Chat/Team/Workflow; create or reopen a session. Send **any message**. In static mode, use the separate explicit action selector and its arguments for addition, report generation, notes, learnings, search or approval. Message phrases do not secretly select deterministic tools. Inspect actual tool/member/step events and saved runs: assistant prose alone is not evidence of execution.

Reload and reopen a session to inspect committed history. To prove restart persistence, record a unique marker and session/run IDs, have the user restart manually, then reopen and compare Oracle evidence. Static history persistence does not establish model comprehension of that history.

For approvals, select/request `publish_report`, observe a persisted pause, then approve. Use a **separate run** to reject. Reopen paused runs and use their actual run IDs; rejected, foreign, mismatched-mode and repeated continuations must not execute an unauthorized tool. This report action is fictional and publishes nothing externally.

Team requires observed Agno member delegation; Workflow requires observed Draft then Review steps. Synthetic outputs can demonstrate those mechanics only. With streaming enabled, **Agent and Team stream genuine Agno content events**. Workflow runs to completion first, then sends its completed response in SSE frames; those frames do not demonstrate incremental workflow/model generation.

Background execution is **API-only, in-process `ThreadPoolExecutor` work**, with an in-memory job list and `durable_queue: false`. It is not an AgentOS durable queue and provides no crash recovery, deduplication or worker fencing. Poll the returned job ID with its owner and submitted request mode; inspect the nested result for effective mode and run status. A job marked completed means the function returned, so inspect whether its Agno run paused or failed. Jobs/polling disappear on restart; committed Agno sessions/runs remain separate Oracle evidence.

Ingest your own named document, inspect embedding provenance, and search as its owner. Static search is a synthetic token-vector exercise; assess semantic paraphrases only with real embeddings. Index creation/refresh is an explicit Oracle mutation. It is not part of opening the page or checking status.

## Manual HTTP examples

All application API routes use `/api`. These loopback examples explicitly request static mode. Replace uppercase response-ID placeholders before use; IDs are returned by the corresponding create/chat/pause/job operation. Inspect mode/provenance and errors in every response. Run each mutation deliberately against fresh owned demo records.

Read-only inspection:

```bash
curl -sS http://127.0.0.1:7777/api/status
curl -sS http://127.0.0.1:7777/api/inventory
curl -sS 'http://127.0.0.1:7777/api/sessions?user_id=alice&mode=static'
curl -sS http://127.0.0.1:7777/api/scenarios
```

Send an arbitrary message, then use the returned session ID to reopen or continue it:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","message":"My unique marker is violet-otter-47."}'
curl -sS 'http://127.0.0.1:7777/api/sessions/SESSION_ID_FROM_RESPONSE?user_id=alice&mode=static'
curl -sS -X POST http://127.0.0.1:7777/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","session_id":"SESSION_ID_FROM_RESPONSE","message":"Here is a follow-up of my own."}'
```

An explicit static action uses separate `action` and `arguments` fields; inspect the real Agno tool result for the computed sum:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","message":"Exercise the addition tool.","action":"add_numbers","arguments":{"a":17,"b":25}}'
```

For genuine Agent streaming, explicitly set `stream: true` and keep curl's output unbuffered. Use `component: "team"` to stream Team events; `component: "workflow"` frames only the completed workflow response:

```bash
curl -N -sS -X POST http://127.0.0.1:7777/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","component":"agent","stream":true,"message":"Inspect streaming provenance."}'
```

Ingest, search and explicitly build/refresh indexes:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/documents \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","name":"Fresh token note","content":"violet-otter-47 identifies this owned demo document."}'
curl -sS -X POST http://127.0.0.1:7777/api/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","query":"violet-otter-47"}'
curl -sS -X POST http://127.0.0.1:7777/api/indexes \
  -H 'Content-Type: application/json' \
  -d '{"mode":"static"}'
```

After the UI/action selector creates a paused `publish_report` run, submit one decision using its returned IDs. For the rejection branch, obtain IDs from a **different paused run** and change `approved` to `false`:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/continue \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","session_id":"PAUSED_SESSION_ID","run_id":"PAUSED_RUN_ID","approved":true}'
```

Submit the same chat-body shape for an in-process background run; streaming is disabled for this path. Use the returned job ID and the same submitted owner/mode when polling:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/background \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","mode":"static","message":"An explicit background integration check."}'
curl -sS 'http://127.0.0.1:7777/api/jobs/JOB_ID_FROM_RESPONSE?user_id=alice&mode=static'
```

`POST /api/checks` implements five **explicit mutating storage checks**, restricted to owned demo tables. Invoke one case at a time; they make no LLM calls and are not startup checks or model-quality tests. The vector-metadata concurrency case requires effective static mode and uses the synthetic embedder only:

```bash
curl -sS -X POST http://127.0.0.1:7777/api/checks \
  -H 'Content-Type: application/json' \
  -d '{"case":"json_reopen","mode":"static"}'
```

Change `case` to request another check, keeping `mode: "static"`:

| Case | Implemented scope |
|---|---|
| `json_reopen` | Write nested learning JSON, read it through a fresh Oracle adapter, then remove that check record. |
| `async_storage` | Native `AsyncOracleDb` write/read/cleanup of nested learning JSON. This does not exercise a native async Agent or async vector search. |
| `concurrency` | Two concurrent vector metadata-key updates, then an explicit null update and cleanup; a static-only smoke check, not exhaustive race proof. |
| `metrics` | Two `calculate_metrics` calls against the selected owned namespace. This refreshes its aggregate metrics. |
| `bulk` | Create 1001 Alice sessions and a Bob control, list with `include_runs=True`, verify owner-scoped deletion, then clean up. No nonempty runs are seeded. |

Component reconstruction, populated-run hydration and further pagination/async-vector branches are separate manual API checks, `NOT RUN`. No command here resets the deployment or deletes old tables. Scheduling, tracing and evaluation APIs are not mounted by this application or run by `/api/checks`.

## Acceptance checklist and pending findings

[`scenarios.py`](scenarios.py) defines 20 areas with `id`, `title`, `steps` (a list), `expected`, `surface`, `static_scope`, `live_scope` and initial `status: NOT RUN`. `GET /api/scenarios` lists that manual catalogue; listing it does not execute it. Record effective mode, provenance, Oracle version, owned tables, session/run IDs, observed errors and scope with each result. Static passes never change live acceptance to passed.

Use this running review checklist to establish understanding before advancing from setup to requests to interpretation:

- [ ] Explain why missing live configuration uses visibly synthetic static mode, while a configured provider error must stay an error.
- [ ] Explain which evidence proves Agno tool execution and committed Oracle state, and why synthetic prose cannot prove live model quality.
- [ ] Verify owner/effective-mode checks and separate embedding identities/tables; explain why the Alice/Bob selector is not authentication.
- [ ] Observe reload/restart behavior, both approval branches and appropriate table/run proof before marking their scenarios.
- [ ] Explain browser/request mode selection, genuine Agent/Team streaming versus completed Workflow frames, and why in-process jobs do not survive restart.
- [ ] Separate the storage checks from pending live, scheduler, tracing, evaluation and compatibility claims; explain how a false pass would mislead later users.

The catalogue covers conversation; table proof/history/restart; sessions/isolation; tools/offload; memory/learnings; approval/rejection; team/workflow; ingestion/vector/text search; concurrent metadata; async storage; background jobs; scheduling; observability; repeated metrics; and bulk/JSON/configuration reconstruction.

Previously observed boundaries were retested; these results do not certify every database version or input size:

| Finding | Required evidence |
|---|---|
| Large report payloads during offload. | **PASS after the 27 October Oracle filesystem fix:** both 500-row (22,702-byte) and 1500-row (68,115-byte) reports produced stored-result references. A fresh runtime recovered every original byte; session deletion removed index and payload rows. The earlier failed runs remain historical evidence and are not backfilled. |
| Repeated metrics refresh previously raised a datetime/date TypeError. | **PASS for this fresh static dataset on Oracle 26.1:** both refresh calls completed. Older-date datasets and other versions remain unverified; no core metrics patch was made. |

Earlier optional-description/evaluation-input constraint findings and API-usage limitations are historical context too; do not describe them as repaired without a targeted retest. No earlier pass counts certify this fresh package, a production identity migration or a real Oracle 19c runtime.

Scheduling, tracing/spans, evaluations and component configuration reconstruction are **not mounted or automatically exercised by this application**. They remain manual Agno API catalogue areas, **NOT RUN** until separately wired/configured and exercised. Saved fixture rows, ordinary chat, job completion or synthetic scores cannot stand in for scheduler execution, linked traces or provider-backed evaluation. Disable any explicitly created schedule after its one-off check.

Static integration results, remaining failures and manual/live-provider acceptance are recorded separately in TEST_LOG.md. This is a loopback, single-process demonstration—not production authentication, durable queue recovery, a full scheduling/tracing/evaluation installation, or proof that every Agno feature works on every supported Oracle version.

## Re-run the checks yourself

In the VM terminal:

```bash
cd /scratch/rodayal/agno-pr-10265
.venv/bin/ruff check e2e_oracle_demo
.venv/bin/ruff format --check e2e_oracle_demo
.venv/bin/python -B -m pytest -q --confcutdir=e2e_oracle_demo -p no:cacheprovider e2e_oracle_demo/test_demo.py
AGNO_DEMO_ALLOW_ORACLE_TESTS=1 .venv/bin/python -B -m pytest -q --confcutdir=e2e_oracle_demo -p no:cacheprovider e2e_oracle_demo/test_oracle_live.py
```

The first pytest command blocks network access and creates no Oracle runtime. The opt-in second command writes synthetic records only into the deployment's owned Oracle namespace and does not call an LLM. These records remain available as demo evidence; the bulk check cleans up only its own generated session IDs. Do not run acceptance tests concurrently with a manual experiment you need to keep isolated.

If loopback HTTP commands on the VM are routed through a corporate proxy, add `--noproxy 127.0.0.1,localhost` to curl. The browser and SSH tunnel do not need that workaround. API keys must stay server-side; see [official OpenAI authentication guidance](https://developers.openai.com/api/reference/overview#authentication).
