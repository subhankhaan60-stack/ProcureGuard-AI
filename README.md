# ProcureGuard AI
Enterprise Procurement Intelligence Platform

[![CI](https://github.com/spalit2025/DocuMatch/actions/workflows/ci.yml/badge.svg)](https://github.com/spalit2025/DocuMatch/actions/workflows/ci.yml)

Privacy-first, 100% offline document processing system that validates invoices against contracts and purchase orders using local LLMs and semantic search.

Procurement teams spend hours manually matching invoices to purchase orders and contracts -- cross-referencing line items, validating amounts, and flagging discrepancies across PDFs that were never designed to talk to each other. DocuMatch automates this with three-way matching (Invoice ↔ PO ↔ Contract), running entirely on your machine.

## Why I Built This

Enterprise procurement matching is a perfect AI automation target: high-volume, rule-heavy, error-prone, and currently done manually in spreadsheets. I wanted to explore three questions:

1. **Can local LLMs handle structured extraction from messy PDFs?** Using Ollama with phi3.5/llama3.2 for entity extraction -- no API costs, no data privacy concerns.
2. **Does semantic search beat keyword matching for document reconciliation?** Vendor names appear differently across documents. Vector similarity handles this naturally.
3. **What does production AI engineering look like?** Not just a demo -- service layer, REST API, batch processing, SQLite state tracking, 238 automated tests.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DocuMatch Architect                           │
├──────────────┬───────────────────────────────────────────────────────┤
│              │                                                       │
│  Streamlit   │   FastAPI REST API                                    │
│  UI (8501)   │   (8000)                                              │
│              │   POST /api/contracts/ingest                          │
│  - Contracts │   POST /api/pos/ingest                                │
│  - POs       │   POST /api/invoices/process                          │
│  - Invoices  │   POST /api/batch/process                             │
│  - Analytics │   GET  /api/batch/{id}/status                         │
│              │   GET  /api/results                                   │
│              │   GET  /api/stats                                     │
│              │   GET  /api/health                                    │
├──────────────┴───────────────────────────────────────────────────────┤
│                        Service Layer                                 │
│  DocumentService (parse→extract→store)                               │
│  MatchService (validate + auto PO matching)                          │
│  BatchService (ThreadPoolExecutor + SQLite state)                    │
├──────────────────────────────────────────────────────────────────────┤
│                        Core Modules                                  │
│  ParserEngine    ExtractionEngine    Matcher    ReportGenerator       │
│  (Docling/       (Ollama LLM)       (Rules +   (Text reports)        │
│   pdfplumber)                        Semantic)                       │
├──────────────────────────────────────────────────────────────────────┤
│                        Data Layer                                    │
│  ChromaDB (contracts + POs)    SQLite (jobs, results, audit log)     │
│  FastEmbed (all-MiniLM-L6-v2) WAL mode for concurrent access        │
└──────────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
Contract:  PDF → Parse (Docling) → Chunk → Index to ChromaDB
PO:        PDF → Parse → LLM Extract → PurchaseOrderSchema → Index to ChromaDB
Invoice:   PDF → Parse → LLM Extract → InvoiceSchema → Three-Way Validation

Three-Way Matching:
  Match 1: Invoice ↔ PO     (amounts, quantities, line items)
  Match 2: Invoice ↔ Contract (rates, dates, payment terms)
  Match 3: PO ↔ Contract     (rates, dates)

  Result: ≥2 matches PASS → PASS | <2 → FAIL/REVIEW
```

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/spalit2025/DocuMatch.git
cd DocuMatch
docker compose up
```

This starts three services:
- **Streamlit UI** at `http://localhost:8501`
- **FastAPI API** at `http://localhost:8000/docs`
- **Ollama** with phi3.5 auto-pulled

### Option 2: Local Development

```bash
# Prerequisites: Python 3.10+, Ollama running locally
ollama pull phi3.5

git clone https://github.com/spalit2025/DocuMatch.git
cd DocuMatch
pip install -r requirements.txt

# Launch UI
streamlit run app/main.py

# Launch API (separate terminal)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## API

Full OpenAPI docs at `http://localhost:8000/docs` when running.

### API access protection

All `/api/*` endpoints require the shared `X-API-Key` header. Configure `API_AUTH_TOKEN` in your `.env` file before using the API. Generate a strong secret locally with:

```bash
openssl rand -hex 32
```

Do not commit the generated secret or share it in screenshots/logs. Docker Compose passes `API_AUTH_TOKEN` from the host environment into the API container. If the token is not configured, API routes fail closed with HTTP 503; missing or incorrect keys receive HTTP 401.

This is a single shared deployment key, **not** individual user accounts, roles, or company-level data isolation. Do not treat it as multi-tenant SaaS authentication.

Example:

```bash
curl -H "X-API-Key: $API_AUTH_TOKEN" http://localhost:8000/api/stats
```

```bash
# Ingest a contract
curl -X POST http://localhost:8000/api/contracts/ingest \
  -F "file=@contract.pdf" \
  -F "vendor_name=Acme Corp" \
  -F "contract_type=MSA"

# Process an invoice (with three-way matching)
curl -X POST http://localhost:8000/api/invoices/process \
  -F "file=@invoice.pdf" \
  -F "po_number=PO-001"

# Batch process multiple invoices
curl -X POST http://localhost:8000/api/batch/process \
  -H "Content-Type: application/json" \
  -d '{"files": [{"file_path": "/data/inv1.pdf"}, {"file_path": "/data/inv2.pdf"}]}'

# Check batch status
curl http://localhost:8000/api/batch/1/status

# Query results
curl "http://localhost:8000/api/results?status=FAIL&vendor_name=Acme"

# System health
curl http://localhost:8000/api/health
```

## Human Review Gate

Every extracted invoice and purchase order starts with `human_review_status: PENDING`.
Extraction and matching results are decision support only; they do not constitute
human approval or authorization to pay. A qualified reviewer must compare critical
values against the original document before approval. The current schema records
the pending state; reviewer identity, approval actions, immutable decision history,
and field-level page citations are not yet implemented.

## Vendor Bank-Detail Change Monitoring

The SQLite metadata store can register a vendor's bank identity and compare later
updates. A changed keyed fingerprint creates a `vendor_bank_details_changed`
audit event and returns `REVIEW`; it is a warning for a human verifier, not a
fraud finding or payment authorization. The stored profile contains the bank
name, account suffix (last four digits), and an opaque fingerprint—not a full
account or routing number.

The caller must calculate `account_fingerprint` as an HMAC over consistently
normalized bank details, using a secret held outside the database and source
control. Do not pass raw account/routing numbers to this method. Plain hashes
are not a substitute for HMAC. The current implementation provides the
database-level registration and change-detection primitive; it does not yet
provide a vendor-facing workflow, reviewer identity, approval/rejection action,
or payment-system block. If the HMAC secret is unavailable or changes, do not
silently register a new baseline—require controlled re-verification.

## Payment Records (Initial Data-Capture Layer)

Payment data is stored separately from invoice validation results. The
authenticated API provides:

- `POST /api/payments` to record a payment entry.
- `GET /api/payments` to list entries, optionally filtered by vendor and status.

New records always start as `PENDING_REVIEW`. The initial layer supports
manual entry or imported payment data, positive amounts with up to two decimal
places, a three-letter currency code, optional payment date/reference, and an
optional link to a previously stored invoice result. When an invoice result is
linked, the vendor and any supplied invoice number must match that result.

The database stores an opaque keyed bank-account fingerprint only when the
caller supplies one; it does not accept or return raw bank account details.
Payment amounts use fixed-precision decimal storage. Each record creates an
audit event.

Example:

```bash
curl -X POST http://localhost:8000/api/payments \\
  -H "X-API-Key: $API_AUTH_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"vendor_name":"Acme Corp","invoice_number":"INV-200","amount":"1250.50","currency":"USD","payment_reference":"BANK-REF-001","source":"MANUAL"}'

curl -H "X-API-Key: $API_AUTH_TOKEN" \\
  "http://localhost:8000/api/payments?vendor_name=Acme%20Corp"
```

## Payment Beneficiary Bank-Profile Checks

The authenticated endpoint `GET /api/risk/payment-bank-mismatches` compares a
payment record's supplied keyed bank-account fingerprint with the currently
registered vendor bank profile. A mismatch is returned as `REVIEW`, with the
registered bank name and account suffix as supporting context. The raw
fingerprints are never returned.

This check only works when both a payment record fingerprint and a registered
vendor baseline exist. Missing fingerprints or missing baselines are not
evidence of a match; those records cannot be compared by this rule. A mismatch
may result from a legitimate bank-detail update or an incorrect/stale source
record. Verify beneficiary details through an independently trusted contact
channel before any payment action.

The endpoint is read-only. It does not change the vendor baseline, block,
approve, execute, or reverse a payment, and a mismatch is not proof of fraud.
The caller remains responsible for generating fingerprints with a keyed HMAC
and protecting the HMAC secret outside the database and source control.

Example:

```bash
curl -H "X-API-Key: $API_AUTH_TOKEN" \\
  "http://localhost:8000/api/risk/payment-bank-mismatches"
```

## Payment Duplicate Review Indicators

The authenticated read-only endpoint `GET /api/risk/payment-duplicates`
checks stored payment records for two candidate patterns:

- The same non-empty payment reference for the same normalized vendor and currency.
- The same normalized vendor, invoice number, amount, and currency appearing in multiple payment records.

The endpoint returns the payment record IDs and supporting fields for human
verification. It does not modify records, block or reverse a transfer, or
conclude that fraud occurred. Repeated invoice payments can be legitimate
installments, and some source systems may reuse payment references. The scan
examines up to the 5,000 most recently recorded payment rows; use a vendor
filter to narrow a review when appropriate.

**Important limitations:** this is payment-data capture only. It does not connect
to a bank, initiate or execute transfers, reconcile bank statements, approve or
reject payments, or enforce segregation of duties. `PENDING_REVIEW` is a
recording status—not proof that a payment is blocked or authorized. Payment
risk rules and a reviewer decision workflow are separate future milestones.

## Evaluation Results

The matching engine is evaluated against 12 synthetic scenarios covering exact matches, partial matches, and full mismatches. Run `python evals/run_evaluation.py` to reproduce.

| Metric | Score |
|--------|-------|
| Status Accuracy | 100% (12/12) |
| Matches Accuracy | 91.7% (11/12) |
| Precision (PASS) | 100% |
| Recall (PASS) | 100% |
| F1 Score | 100% |

| Category | Scenarios | Accuracy |
|----------|-----------|----------|
| Exact Match | 7 | 100% |
| Partial Match | 2 | 100% |
| Full Mismatch | 3 | 100% |

## Key Features

| Feature | Description |
|---------|-------------|
| Three-Way Matching | Invoice ↔ PO ↔ Contract validation with configurable tolerance |
| Auto PO Matching | Automatically finds the matching PO by number or vendor+amount fuzzy match |
| Batch Processing | Process 100 invoices concurrently with per-file error isolation |
| REST API | 8 endpoints with OpenAPI docs, proper error codes (422/502/503) |
| Analytics Dashboard | KPI cards, Plotly charts, filterable results table |
| Audit Trail | Every action logged to SQLite with timestamps |\n| Payment Records | Separate payment-data register with review status and invoice linkage |
| 100% Offline | No data leaves your machine -- local LLMs + local vector DB |

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| UI | Streamlit | Rapid prototyping, built-in file upload, session state |
| API | FastAPI | Auto OpenAPI docs, dependency injection, async support |
| LLM | Ollama (phi3.5) | Local inference, no API costs, privacy-first |
| Vector DB | ChromaDB + FastEmbed | Semantic search with all-MiniLM-L6-v2 embeddings |
| PDF Parsing | Docling + pdfplumber | Dual parser with automatic fallback |
| Database | SQLite + SQLAlchemy | WAL mode for concurrent access, zero config |
| Validation | Pydantic v2 | Type-safe schemas with field validators |
| Charts | Plotly | Interactive, publication-quality visualizations |
| Testing | pytest (238 tests) | Unit, API, hostile input, integration tests |
| CI | GitHub Actions | Lint + test on Python 3.11/3.12 |
| Deploy | Docker Compose | One command: `docker compose up` |

## Testing

```bash
# Run all tests (excluding integration tests that need Ollama)
python -m pytest tests/ -v -k "not Integration and not real_extraction"

# Run specific test suites
python -m pytest tests/test_api.py -v              # API contract tests
python -m pytest tests/test_hostile_inputs.py -v    # Adversarial inputs
python -m pytest tests/test_batch_service.py -v     # Batch processing
python -m pytest tests/test_database.py -v          # SQLite operations
python -m pytest tests/test_document_service.py -v  # Service layer
python -m pytest tests/test_match_service.py -v     # Matching + auto PO
```

## Configuration

All settings via `.env` or environment variables:

```bash
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=phi3.5
CHROMA_PERSIST_DIR=./data/chroma_db
DB_PATH=./data/documatch.db
MATCH_TOLERANCE=0.01           # 1% amount tolerance
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=512
CHUNK_OVERLAP=50
LOG_LEVEL=INFO
```

## Project Structure

```
DocuMatch/
├── api/                    # FastAPI REST API
│   ├── app.py              # App creation, exception handlers
│   ├── routes.py           # Endpoint handlers
│   ├── schemas.py          # API request/response models
│   └── dependencies.py     # Dependency injection
├── app/                    # Streamlit UI
│   ├── main.py             # Entry point
│   └── pages/              # Multi-page app
│       ├── 1_Ingest_Contracts.py
│       ├── 2_Process_POs.py
│       ├── 3_Process_Invoices.py
│       └── 4_Analytics.py
├── core/                   # Business logic
│   ├── services/           # Orchestration layer
│   │   ├── document_service.py
│   │   ├── match_service.py
│   │   └── batch_service.py
│   ├── parser_engine.py    # PDF → Markdown
│   ├── extraction.py       # LLM extraction
│   ├── matcher.py          # Validation rules
│   ├── vector_store.py     # ChromaDB operations
│   ├── po_store.py         # PO storage
│   ├── database.py         # SQLite metadata store
│   └── models.py           # Pydantic schemas
├── prompts/                # LLM prompt templates
├── tests/                  # 238 tests
├── docker-compose.yml      # One-command deployment
├── Dockerfile
└── requirements.txt
```

## License

MIT


## Procurement Risk Indicators

The authenticated API endpoint `GET /api/risk/invoice-patterns` identifies repeated same-vendor, same-amount, same-currency invoice references in a configurable processing-time window. It returns the supporting result IDs and invoice references for human review.

This is an invoice-pattern indicator only. The current data model does not store actual payment transactions, payment dates, payment approvals, or payment execution records; therefore this feature does not claim to detect suspicious payments. Repeated amounts can be legitimate recurring billing. The default window is 7 days and the default minimum is 3 distinct invoice references; both can be configured in the request. The window is based on result processing timestamps, not extracted invoice dates.



### Validation explanations and evidence

- Stored invoice validation results now include a deterministic severity summary and the serialized validation issues that support it.
- Matched contract clause snippets and their existing metadata are retained where the matcher supplies them.
- Retrieve a saved result's explanation and evidence with the authenticated endpoint `GET /api/results/{result_id}/evidence`.
- The extraction pipeline does not yet retain reliable source page numbers, so the feature does not claim page-level traceability. Evidence is limited to data returned by the current parser and matcher.
- A risk explanation is an aid to human review, not a fraud determination or payment approval.



## Payment Human Review Dispositions

The authenticated API supports a one-time human review disposition for a payment
record that is still `PENDING_REVIEW`:

- `POST /api/payments/{payment_id}/review` with `decision: CLEAR` or `REJECT`,
  a reviewer reference, and a non-empty explanation.
- The payment record becomes `REVIEWED_CLEAR` or `REVIEWED_REJECTED`.
- `GET /api/payments/{payment_id}/reviews` returns the recorded review history.
- Each decision creates a separate review-event row and an audit-log event.
  A second decision cannot overwrite the first through this endpoint.

A `CLEAR` disposition means only that the supplied reviewer marked the record
clear at this review stage. It is **not** a bank instruction, payment execution,
or proof that all required organizational approvals have been obtained.
The current shared API-key authentication does not establish an individual's
identity or enforce segregation of duties; `reviewer_reference` is caller-
supplied metadata and must not be treated as verified identity. Production use
requires individual authenticated accounts, role-based permissions, and
independent approval controls.
