"""
API tests for DocuMatch Architect.

Uses FastAPI TestClient with mocked services to test:
- Request/response contracts for all 4 endpoints
- Error handling (422, 502, 503)
- File upload handling
- Health check component status

All core modules are mocked -- we test the API layer, not business logic.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_batch_service, get_database, get_document_service, get_match_service, get_current_user
from core.exceptions import StoreError
from core.extraction import ExtractionError
from core.models import (
    InvoiceSchema,
    LineItem,
    MatchDetail,
    MatchResult,
    ParseResult,
    PurchaseOrderSchema,
    ThreeWayMatchResult,
    ValidationIssue,
)
from core.services import DocumentProcessingError
from config import settings


# ==================== FIXTURES ====================


@pytest.fixture
def mock_doc_service():
    service = MagicMock()

    # Default: parser available, extraction engine connected
    service.parser._docling_available = True
    service.extraction_engine.check_connection.return_value = (True, "Connected. Model 'phi3.5' available.")
    service.vector_store.get_stats.return_value = {"total_chunks": 10, "total_vendors": 2}

    return service


@pytest.fixture
def mock_match_service():
    return MagicMock()


@pytest.fixture
def mock_batch_service():
    return MagicMock()


@pytest.fixture
def mock_database():
    return MagicMock()


@pytest.fixture
def client(mock_doc_service, mock_match_service, mock_batch_service, mock_database, monkeypatch):
    """TestClient with mocked dependencies."""
    monkeypatch.setattr(settings, "api_auth_token", "test-secret")
    app.dependency_overrides[get_document_service] = lambda: mock_doc_service
    app.dependency_overrides[get_match_service] = lambda: mock_match_service
    app.dependency_overrides[get_batch_service] = lambda: mock_batch_service
    app.dependency_overrides[get_database] = lambda: mock_database
    app.dependency_overrides[get_current_user] = lambda: {"id": 1, "company_id": 1, "email": "admin@example.com", "full_name": "Test Admin", "role": "ADMIN", "is_active": True}
    yield TestClient(app, headers={"X-API-Key": "test-secret"})
    app.dependency_overrides.clear()


@pytest.fixture
def fake_pdf():
    """A fake PDF file for upload testing."""
    return ("test.pdf", io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")


@pytest.fixture
def sample_parse_result():
    return ParseResult(
        markdown="# Document\nContent here",
        page_count=3,
        tables_found=1,
        parse_method="docling",
        success=True,
    )


@pytest.fixture
def sample_invoice():
    return InvoiceSchema(
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        invoice_date="2024-01-15",
        total_amount=5000.00,
        currency="USD",
        payment_terms="Net 30",
        po_number="PO-001",
        line_items=[
            LineItem(description="Consulting", quantity=40, unit_price=125.00, total=5000.00)
        ],
    )


@pytest.fixture
def sample_po():
    return PurchaseOrderSchema(
        po_number="PO-001",
        vendor_name="Acme Corp",
        order_date="2024-01-10",
        total_amount=5000.00,
        currency="USD",
        line_items=[
            LineItem(description="Consulting", quantity=40, unit_price=125.00, total=5000.00)
        ],
    )


@pytest.fixture
def sample_three_way_result():
    return ThreeWayMatchResult(
        status="PASS",
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        po_number="PO-001",
        invoice_po_match=MatchDetail(
            match_type="invoice_po", passed=True, score=0.95,
            issues=[],
        ),
        invoice_contract_match=MatchDetail(
            match_type="invoice_contract", passed=True, score=0.90,
            issues=[
                ValidationIssue(
                    rule="rate_card_found", severity="info",
                    message="Could not find specific rates",
                    invoice_value=None, match_type="invoice_contract",
                )
            ],
        ),
        po_contract_match=None,
        matches_passed=2,
        total_matches=2,
        overall_score=0.925,
        all_issues=[
            ValidationIssue(
                rule="rate_card_found", severity="info",
                message="Could not find specific rates",
                invoice_value=None, match_type="invoice_contract",
            )
        ],
        matched_clauses=[],
    )


# ==================== API AUTHENTICATION ====================

class TestAPIAuthentication:
    def test_rejects_missing_api_key(self, client):
        response = client.get("/api/stats", headers={"X-API-Key": ""})
        assert response.status_code == 401

    def test_rejects_invalid_api_key(self, client):
        response = client.get("/api/stats", headers={"X-API-Key": "wrong-secret"})
        assert response.status_code == 401

    def test_rejects_requests_when_server_secret_is_not_configured(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_auth_token", None)
        response = client.get("/api/stats")
        assert response.status_code == 503
        assert "API_AUTH_TOKEN" in response.json()["detail"]


# ==================== CONTRACT INGESTION ====================


class TestContractIngest:
    """Tests for POST /api/contracts/ingest."""

    def test_success(self, client, mock_doc_service, fake_pdf, sample_parse_result):
        mock_doc_service.ingest_contract.return_value = ("contract-abc", sample_parse_result)

        response = client.post(
            "/api/contracts/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme Corp", "contract_type": "MSA"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["contract_id"] == "contract-abc"
        assert data["vendor_name"] == "Acme Corp"
        assert data["contract_type"] == "MSA"
        assert data["parse_info"]["page_count"] == 3
        assert data["parse_info"]["parse_method"] == "docling"

    def test_default_contract_type(self, client, mock_doc_service, fake_pdf, sample_parse_result):
        mock_doc_service.ingest_contract.return_value = ("contract-abc", sample_parse_result)

        response = client.post(
            "/api/contracts/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 200
        assert response.json()["contract_type"] == "MSA"

    def test_parse_failure_returns_422(self, client, mock_doc_service, fake_pdf):
        mock_doc_service.ingest_contract.side_effect = DocumentProcessingError(
            "Failed to parse test.pdf: Corrupt PDF"
        )

        response = client.post(
            "/api/contracts/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "document_processing_error"
        assert "Corrupt PDF" in response.json()["detail"]

    def test_store_error_returns_503(self, client, mock_doc_service, fake_pdf):
        mock_doc_service.ingest_contract.side_effect = StoreError("ChromaDB down")

        response = client.post(
            "/api/contracts/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 503
        assert response.json()["error"] == "store_error"

    def test_missing_vendor_name_returns_422(self, client, fake_pdf):
        response = client.post(
            "/api/contracts/ingest",
            files={"file": fake_pdf},
        )

        assert response.status_code == 422

    def test_missing_file_returns_422(self, client):
        response = client.post(
            "/api/contracts/ingest",
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 422


# ==================== PO INGESTION ====================


class TestPOIngest:
    """Tests for POST /api/pos/ingest."""

    def test_success(self, client, mock_doc_service, fake_pdf, sample_po, sample_parse_result):
        mock_doc_service.ingest_po.return_value = (sample_po, sample_parse_result)

        response = client.post(
            "/api/pos/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme Corp"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["po_number"] == "PO-001"
        assert data["vendor_name"] == "Acme Corp"
        assert data["total_amount"] == 5000.00
        assert len(data["line_items"]) == 1
        assert data["line_items"][0]["description"] == "Consulting"
        assert data["parse_info"]["page_count"] == 3

    def test_extraction_error_returns_502(self, client, mock_doc_service, fake_pdf):
        mock_doc_service.ingest_po.side_effect = ExtractionError(
            "Request timed out after 120s"
        )

        response = client.post(
            "/api/pos/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 502
        assert response.json()["error"] == "extraction_error"
        assert "timed out" in response.json()["detail"]

    def test_parse_failure_returns_422(self, client, mock_doc_service, fake_pdf):
        mock_doc_service.ingest_po.side_effect = DocumentProcessingError(
            "File too large: 75.0MB exceeds limit"
        )

        response = client.post(
            "/api/pos/ingest",
            files={"file": fake_pdf},
            data={"vendor_name": "Acme"},
        )

        assert response.status_code == 422


# ==================== INVOICE PROCESSING ====================


class TestInvoiceProcess:
    """Tests for POST /api/invoices/process."""

    def test_success_with_po(
        self, client, mock_doc_service, mock_match_service,
        fake_pdf, sample_invoice, sample_parse_result, sample_three_way_result,
    ):
        mock_doc_service.process_invoice.return_value = (sample_invoice, sample_parse_result)
        mock_match_service.validate_three_way.return_value = sample_three_way_result

        response = client.post(
            "/api/invoices/process",
            files={"file": fake_pdf},
            data={"po_number": "PO-001"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["invoice_number"] == "INV-001"
        assert data["vendor_name"] == "Acme Corp"
        assert data["total_amount"] == 5000.00
        assert data["po_number"] == "PO-001"

        # Validation section
        v = data["validation"]
        assert v["status"] == "PASS"
        assert v["matches_passed"] == 2
        assert v["total_matches"] == 2
        assert v["invoice_po_match"]["passed"] is True
        assert v["invoice_contract_match"]["passed"] is True
        assert v["po_contract_match"] is None
        assert len(v["issues"]) == 1

    def test_success_without_po(
        self, client, mock_doc_service, mock_match_service,
        fake_pdf, sample_invoice, sample_parse_result, sample_three_way_result,
    ):
        mock_doc_service.process_invoice.return_value = (sample_invoice, sample_parse_result)
        mock_match_service.validate_three_way.return_value = sample_three_way_result

        response = client.post(
            "/api/invoices/process",
            files={"file": fake_pdf},
        )

        assert response.status_code == 200
        # po_number should come from invoice's po_number field
        assert response.json()["po_number"] == "PO-001"

    def test_extraction_error_returns_502(self, client, mock_doc_service, fake_pdf):
        mock_doc_service.process_invoice.side_effect = ExtractionError(
            "Cannot connect to Ollama. Is it running?"
        )

        response = client.post(
            "/api/invoices/process",
            files={"file": fake_pdf},
        )

        assert response.status_code == 502
        assert "Ollama" in response.json()["detail"]

    def test_validation_fail_result(
        self, client, mock_doc_service, mock_match_service,
        fake_pdf, sample_invoice, sample_parse_result,
    ):
        mock_doc_service.process_invoice.return_value = (sample_invoice, sample_parse_result)
        mock_match_service.validate_three_way.return_value = ThreeWayMatchResult(
            status="FAIL",
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            po_number=None,
            invoice_po_match=None,
            invoice_contract_match=MatchDetail(
                match_type="invoice_contract", passed=False, score=0.2,
                issues=[
                    ValidationIssue(
                        rule="contract_exists", severity="critical",
                        message="No contract found for vendor 'Acme Corp'",
                        invoice_value="Acme Corp",
                        match_type="invoice_contract",
                    )
                ],
            ),
            po_contract_match=None,
            matches_passed=0,
            total_matches=1,
            overall_score=0.2,
            all_issues=[
                ValidationIssue(
                    rule="contract_exists", severity="critical",
                    message="No contract found for vendor 'Acme Corp'",
                    invoice_value="Acme Corp",
                    match_type="invoice_contract",
                )
            ],
            matched_clauses=[],
        )

        response = client.post(
            "/api/invoices/process",
            files={"file": fake_pdf},
        )

        assert response.status_code == 200  # HTTP 200, but validation FAIL
        v = response.json()["validation"]
        assert v["status"] == "FAIL"
        assert v["matches_passed"] == 0
        assert v["issues"][0]["severity"] == "critical"


class TestPaymentsAPI:
    def test_payment_api_rejects_user_without_company_assignment(self, client, monkeypatch):
        from api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = lambda: {
            "id": 9, "company_id": None, "email": "unassigned@example.com",
            "full_name": "Unassigned", "role": "ADMIN", "is_active": True,
        }
        response = client.get("/api/payments")
        assert response.status_code == 403

    def test_create_payment_records_pending_review(self, client, mock_database):
        mock_database.create_payment_record.return_value = {
            "id": 7,
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-200",
            "invoice_result_id": 12,
            "payment_reference": "BANK-REF-001",
            "amount": "1250.50",
            "currency": "USD",
            "payment_date": "2026-11-08",
            "source": "MANUAL",
            "status": "PENDING_REVIEW",
            "created_by_user_id": 1,
            "created_at": "2026-11-08T12:00:00+00:00",
        }

        response = client.post(
            "/api/payments",
            json={
                "vendor_name": "Acme Corp",
                "invoice_result_id": 12,
                "payment_reference": "BANK-REF-001",
                "amount": "1250.50",
                "currency": "usd",
                "payment_date": "2026-11-08",
                # A client-supplied creator ID must never override authenticated identity.
                "created_by_user_id": 999,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PENDING_REVIEW"
        assert data["amount"] == "1250.50"
        assert data["currency"] == "USD"
        assert data["created_by_user_id"] == 1
        assert "bank_account_fingerprint" not in data
        mock_database.create_payment_record.assert_called_once()
        assert mock_database.create_payment_record.call_args.kwargs["created_by_user_id"] == 1
        assert mock_database.create_payment_record.call_args.kwargs["company_id"] == 1

    def test_create_payment_rejects_invalid_amount(self, client):
        response = client.post(
            "/api/payments",
            json={"vendor_name": "Acme Corp", "amount": "0", "currency": "USD"},
        )
        assert response.status_code == 422

    def test_create_payment_maps_link_validation_to_422(self, client, mock_database):
        mock_database.create_payment_record.side_effect = ValueError(
            "Linked invoice result was not found."
        )
        response = client.post(
            "/api/payments",
            json={
                "vendor_name": "Acme Corp",
                "invoice_result_id": 999,
                "amount": "100.00",
                "currency": "USD",
            },
        )
        assert response.status_code == 422

    def test_list_payments(self, client, mock_database):
        mock_database.list_payment_records.return_value = [{
            "id": 1,
            "vendor_name": "Acme Corp",
            "invoice_number": None,
            "invoice_result_id": None,
            "payment_reference": None,
            "amount": "100.00",
            "currency": "USD",
            "payment_date": None,
            "source": "MANUAL",
            "status": "PENDING_REVIEW",
            "created_at": "2026-11-08T12:00:00+00:00",
        }]
        response = client.get("/api/payments?vendor_name=Acme&limit=10")
        assert response.status_code == 200
        assert response.json()[0]["status"] == "PENDING_REVIEW"
        mock_database.list_payment_records.assert_called_once_with(
            vendor_name="Acme", status=None, limit=10, company_id=1,
        )


    def test_payment_review_endpoint_records_disposition(self, client, mock_database):
        mock_database.review_payment_record.return_value = {
            "payment": {"id": 5, "status": "REVIEWED_CLEAR"},
            "review": {"id": 1, "payment_id": 5, "decision": "CLEAR"},
            "payment_execution_authorized": False,
        }
        response = client.post("/api/payments/5/review", json={
            "decision": "CLEAR",
            "note": "Checked supporting documents.",
        })
        assert response.status_code == 200
        assert response.json()["payment_execution_authorized"] is False
        mock_database.review_payment_record.assert_called_once_with(
            payment_id=5,
            decision="CLEAR",
            reviewer_reference="admin@example.com",
            note="Checked supporting documents.",
            reviewer_user_id=1,
            company_id=1,
        )

    def test_payment_creator_cannot_review_own_payment(self, client, mock_database):
        mock_database.review_payment_record.side_effect = ValueError(
            "Payment creator cannot review their own payment (segregation of duties)."
        )
        response = client.post("/api/payments/5/review", json={
            "decision": "CLEAR",
            "note": "Attempted self-review.",
        })
        assert response.status_code == 409
        mock_database.review_payment_record.assert_called_once_with(
            payment_id=5,
            decision="CLEAR",
            reviewer_reference="admin@example.com",
            note="Attempted self-review.",
            reviewer_user_id=1,
        )


    def test_payment_duplicate_risk_endpoint_returns_review_indicators(self, client, mock_database):
        mock_database.find_duplicate_payment_candidates.return_value = [{
            "rule": "duplicate_payment_reference",
            "status": "REVIEW",
            "vendor_name": "Acme Corp",
            "occurrences": 2,
            "payment_ids": [1, 2],
            "evidence": [{"payment_id": 1}, {"payment_id": 2}],
            "explanation": "Verify the repeated payment reference.",
        }]
        response = client.get("/api/risk/payment-duplicates?vendor_name=Acme&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "payment_duplicate_review"
        assert data["finding_count"] == 1
        assert data["findings"][0]["status"] == "REVIEW"
        assert "not blocked" in data["disclaimer"]
        mock_database.find_duplicate_payment_candidates.assert_called_once_with(
            vendor_name="Acme", limit=10, company_id=1,
        )

    def test_payment_duplicate_risk_endpoint_maps_invalid_filter_to_422(self, client, mock_database):
        mock_database.find_duplicate_payment_candidates.side_effect = ValueError(
            "limit must be between 1 and 500."
        )
        response = client.get("/api/risk/payment-duplicates?limit=0")
        assert response.status_code == 422


    def test_payment_bank_mismatch_endpoint_returns_review_indicator(self, client, mock_database):
        mock_database.find_payment_bank_profile_mismatches.return_value = [{
            "rule": "payment_bank_profile_mismatch",
            "status": "REVIEW",
            "payment_id": 9,
            "vendor_name": "Acme Corp",
            "registered_bank_name": "Example Bank",
            "registered_account_last4": "1234",
            "payment_authorized": False,
        }]
        response = client.get(
            "/api/risk/payment-bank-mismatches?vendor_name=Acme&limit=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "payment_bank_profile_mismatch_review"
        assert data["finding_count"] == 1
        assert data["findings"][0]["status"] == "REVIEW"
        assert "does not block" in data["disclaimer"]
        mock_database.find_payment_bank_profile_mismatches.assert_called_once_with(
            vendor_name="Acme", limit=5, company_id=1,
        )

    def test_payment_bank_mismatch_endpoint_maps_invalid_limit_to_422(
        self, client, mock_database
    ):
        response = client.get("/api/risk/payment-bank-mismatches?limit=0")
        assert response.status_code == 422
        mock_database.find_payment_bank_profile_mismatches.assert_not_called()


    def test_payment_review_queue_combines_pending_records_and_risk_flags(
        self, client, mock_database
    ):
        mock_database.list_payment_records.return_value = [
            {"id": 1, "vendor_name": "Acme Corp", "status": "PENDING_REVIEW"},
            {"id": 2, "vendor_name": "Other Corp", "status": "PENDING_REVIEW"},
        ]
        mock_database.find_duplicate_payment_candidates.return_value = [
            {"rule": "duplicate_payment_reference", "payment_ids": [1, 3]}
        ]
        mock_database.find_payment_bank_profile_mismatches.return_value = [
            {"rule": "payment_bank_profile_mismatch", "payment_id": 2}
        ]

        response = client.get("/api/risk/payment-review-queue?limit=20")

        assert response.status_code == 200
        data = response.json()
        assert data["record_count"] == 2
        assert data["flagged_count"] == 2
        assert data["records"][0]["risk_flags"] == ["duplicate_payment_reference"]
        assert data["records"][0]["risk_review_required"] is True
        assert data["records"][1]["risk_flags"] == ["payment_bank_profile_mismatch"]
        assert data["records"][1]["risk_review_required"] is True
        assert "does not approve" in data["disclaimer"]
        mock_database.list_payment_records.assert_called_once_with(
            vendor_name=None, status="PENDING_REVIEW", limit=20, company_id=1
        )

    def test_payment_review_queue_marks_unflagged_pending_records(
        self, client, mock_database
    ):
        mock_database.list_payment_records.return_value = [
            {"id": 8, "vendor_name": "Vendor Z", "status": "PENDING_REVIEW"}
        ]
        mock_database.find_duplicate_payment_candidates.return_value = []
        mock_database.find_payment_bank_profile_mismatches.return_value = []

        response = client.get("/api/risk/payment-review-queue")

        assert response.status_code == 200
        record = response.json()["records"][0]
        assert record["risk_flags"] == []
        assert record["risk_review_required"] is False

    def test_payment_review_queue_maps_database_validation_errors_to_422(
        self, client, mock_database
    ):
        mock_database.list_payment_records.side_effect = ValueError(
            "limit must be between 1 and 500."
        )
        response = client.get("/api/risk/payment-review-queue?limit=0")
        assert response.status_code == 422


# ==================== HEALTH CHECK ====================


class TestHealthCheck:
    """Tests for GET /api/health."""

    def test_cors_allows_configured_local_origin(self, client):
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type,x-api-key",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "x-api-key" in response.headers["access-control-allow-headers"].lower()
        assert "access-control-allow-credentials" not in response.headers

    def test_cors_rejects_unconfigured_origin(self, client):
        response = client.options(
            "/api/health",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert "access-control-allow-origin" not in response.headers

    def test_healthy(self, client, mock_doc_service):
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["parser"]["status"] == "ok"
        assert data["components"]["ollama"]["status"] == "ok"
        assert data["components"]["chromadb"]["status"] == "ok"

    def test_degraded_ollama_down(self, client, mock_doc_service):
        mock_doc_service.extraction_engine.check_connection.return_value = (
            False, "Ollama not running"
        )

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["ollama"]["status"] == "error"
        assert data["components"]["parser"]["status"] == "ok"

    def test_degraded_chromadb_down(self, client, mock_doc_service):
        mock_doc_service.vector_store.get_stats.side_effect = Exception("Connection refused")

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["chromadb"]["status"] == "error"
        assert data["components"]["chromadb"]["detail"] == "Health check failed"

    def test_pdfplumber_fallback(self, client, mock_doc_service):
        mock_doc_service.parser._docling_available = False

        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["components"]["parser"]["detail"] == "pdfplumber"


# ==================== BATCH PROCESSING ====================


class TestBatchAPI:
    """Tests for batch processing API endpoints."""

    def test_submit_batch(self, client, mock_batch_service, tmp_path, monkeypatch):
        mock_batch_service.submit_batch.return_value = 42
        monkeypatch.setattr(settings, "invoices_dir", str(tmp_path))

        invoice_one = tmp_path / "inv1.pdf"
        invoice_two = tmp_path / "inv2.pdf"
        invoice_one.write_bytes(b"%PDF-1.4 fake content")
        invoice_two.write_bytes(b"%PDF-1.4 fake content")

        response = client.post(
            "/api/batch/process",
            json={
                "files": [
                    {"file_path": str(invoice_one), "po_number": "PO-001"},
                    {"file_path": str(invoice_two)},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == 42
        assert data["total_files"] == 2
        assert data["status"] == "PARSING"

    def test_submit_empty_batch(self, client):
        response = client.post(
            "/api/batch/process",
            json={"files": []},
        )
        assert response.status_code == 422

    def test_get_batch_status(self, client, mock_batch_service):
        from core.services.batch_service import BatchStatus
        mock_batch_service.get_batch_status.return_value = BatchStatus(
            job_id=42,
            status="PARSING",
            total_files=3,
            completed=1,
            failed=0,
            pending=1,
            processing=1,
            eta_seconds=30.5,
            errors=[],
        )

        response = client.get("/api/batch/42/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == 42
        assert data["total_files"] == 3
        assert data["completed"] == 1
        assert data["eta_seconds"] == 30.5

    def test_get_batch_status_not_found(self, client, mock_batch_service):
        mock_batch_service.get_batch_status.return_value = None

        response = client.get("/api/batch/999/status")
        assert response.status_code == 404

    def test_cancel_batch(self, client, mock_batch_service):
        mock_batch_service.cancel_batch.return_value = True

        response = client.post("/api/batch/42/cancel")

        assert response.status_code == 200
        assert "cancellation" in response.json()["message"]

    def test_cancel_batch_not_found(self, client, mock_batch_service):
        mock_batch_service.cancel_batch.return_value = False

        response = client.post("/api/batch/999/cancel")
        assert response.status_code == 404


# ==================== RESULTS & STATS ====================


class TestResultsAPI:
    """Tests for GET /api/results and GET /api/stats."""

    def test_get_results_empty(self, client, mock_database):
        mock_database.get_results.return_value = []

        response = client.get("/api/results")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_results_with_data(self, client, mock_database):
        from datetime import datetime, timezone
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.job_id = 10
        mock_result.invoice_file = "invoice.pdf"
        mock_result.vendor_name = "Acme"
        mock_result.invoice_number = "INV-001"
        mock_result.status = "PASS"
        mock_result.confidence = 0.95
        mock_result.matches_passed = 2
        mock_result.total_matches = 2
        mock_result.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        mock_database.get_results.return_value = [mock_result]

        response = client.get("/api/results")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["invoice_number"] == "INV-001"
        assert data[0]["status"] == "PASS"

    def test_get_results_with_filters(self, client, mock_database):
        mock_database.get_results.return_value = []

        client.get("/api/results?vendor_name=Acme&status=PASS&limit=10")

        mock_database.get_results.assert_called_once_with(
            vendor_name="Acme", status="PASS", job_id=None, limit=10,
        )

    def test_get_stats(self, client, mock_database):
        mock_database.get_stats.return_value = {
            "total_jobs": 10,
            "completed_jobs": 8,
            "failed_jobs": 1,
            "pending_jobs": 1,
            "total_results": 15,
            "pass_count": 12,
            "fail_count": 2,
            "review_count": 1,
            "pass_rate": 0.8,
        }

        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_jobs"] == 10
        assert data["pass_rate"] == 0.8
        assert data["pass_count"] == 12


class TestCompanyProvisioningAPI:
    def test_admin_can_create_company(self, client, mock_database):
        mock_database.create_company.return_value = {
            "id": 12, "name": "Apex Global", "slug": "apex-global",
            "is_active": True, "created_at": None,
        }
        response = client.post("/api/companies", json={
            "name": "Apex Global", "slug": "apex-global",
        })
        assert response.status_code == 200
        assert response.json()["id"] == 12
        mock_database.create_company.assert_called_once_with(
            name="Apex Global", slug="apex-global"
        )

    def test_company_creation_maps_validation_errors(self, client, mock_database):
        mock_database.create_company.side_effect = ValueError("Company slug is invalid.")
        response = client.post("/api/companies", json={
            "name": "Apex Global", "slug": "apex-global",
        })
        assert response.status_code == 422

    def test_user_provisioning_accepts_company_assignment(self, client, mock_database):
        mock_database.create_user.return_value = {
            "id": 13, "company_id": 12, "email": "member@example.com",
            "full_name": "Company Member", "role": "REVIEWER",
            "is_active": True, "created_at": None,
        }
        response = client.post("/api/auth/users", json={
            "email": "member@example.com", "full_name": "Company Member",
            "password": "a-strong-password-123", "role": "REVIEWER",
            "company_id": 12,
        })
        assert response.status_code == 200
        assert response.json()["company_id"] == 12
        mock_database.create_user.assert_called_once()
        assert mock_database.create_user.call_args.kwargs["company_id"] == 12


class TestUserLoginAPI:
    def test_login_returns_short_lived_bearer_token(self, client, mock_database, monkeypatch):
        mock_database.login_rate_limit_remaining.return_value = 0
        monkeypatch.setattr(settings, "user_session_secret", "test-session-secret-that-is-at-least-32-bytes")
        monkeypatch.setattr(settings, "user_session_ttl_seconds", 1800)
        mock_database.authenticate_user.return_value = {
            "id": 3, "email": "reviewer@example.com", "full_name": "Reviewer",
            "role": "REVIEWER", "is_active": True, "created_at": None,
        }
        response = client.post("/api/auth/login", json={
            "email": "reviewer@example.com", "password": "correct-password"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 1800
        assert data["user"]["role"] == "REVIEWER"
        assert "password" not in data["user"]
        assert data["access_token"]

    def test_login_throttles_after_five_failures(self, client, mock_database):
        mock_database.login_rate_limit_remaining.return_value = 900
        response = client.post("/api/auth/login", json={
            "email": "reviewer@example.com", "password": "wrong-password"
        })
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "900"
        mock_database.authenticate_user.assert_not_called()

    def test_invalid_login_records_failure(self, client, mock_database):
        mock_database.login_rate_limit_remaining.return_value = 0
        mock_database.authenticate_user.return_value = None
        response = client.post("/api/auth/login", json={
            "email": "reviewer@example.com", "password": "wrong-password"
        })
        assert response.status_code == 401
        mock_database.record_failed_login.assert_called_once_with(
            "reviewer@example.com", "testclient"
        )

    def test_login_rejects_invalid_credentials(self, client, mock_database, monkeypatch):
        mock_database.login_rate_limit_remaining.return_value = 0
        monkeypatch.setattr(settings, "user_session_secret", "test-session-secret-that-is-at-least-32-bytes")
        mock_database.authenticate_user.return_value = None
        response = client.post("/api/auth/login", json={
            "email": "reviewer@example.com", "password": "wrong-password"
        })
        assert response.status_code == 401

    def test_login_requires_session_secret(self, client, mock_database, monkeypatch):
        mock_database.login_rate_limit_remaining.return_value = 0
        monkeypatch.setattr(settings, "user_session_secret", None)
        mock_database.authenticate_user.return_value = {
            "id": 3, "email": "reviewer@example.com", "full_name": "Reviewer",
            "role": "REVIEWER", "is_active": True,
        }
        response = client.post("/api/auth/login", json={
            "email": "reviewer@example.com", "password": "correct-password"
        })
        assert response.status_code == 503


    def test_admin_can_create_user_account(self, client, mock_database):
        mock_database.create_user.return_value = {
            "id": 8, "email": "preparer@example.com", "full_name": "Preparer",
            "role": "PREPARER", "is_active": True, "created_at": None,
        }
        response = client.post("/api/auth/users", json={
            "email": "preparer@example.com", "full_name": "Preparer",
            "password": "a-strong-password-123", "role": "PREPARER",
        })
        assert response.status_code == 200
        assert response.json()["role"] == "PREPARER"
        assert "password_hash" not in response.json()
        mock_database.create_user.assert_called_once()
