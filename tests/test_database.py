"""
Tests for SQLite Metadata Store.

Tests all database operations: jobs, results, audit log, and stats.
Uses temp directories so each test gets a fresh database.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from pathlib import Path

from core.database import AuditLog, Base, Database, Job, Result


# ==================== FIXTURES ====================


@pytest.fixture
def db(tmp_path):
    """Create a fresh database for each test."""
    db_path = str(tmp_path / "test.db")
    return Database(db_path)


# ==================== DATABASE INIT ====================


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        db = Database(db_path)
        assert Path(db_path).exists()

    def test_creates_parent_directories(self, tmp_path):
        db_path = str(tmp_path / "nested" / "dir" / "test.db")
        db = Database(db_path)
        assert Path(db_path).exists()

    def test_wal_mode_enabled(self, db):
        """Verify WAL mode is active for concurrent access."""
        with db.engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text("PRAGMA journal_mode")
            )
            mode = result.scalar()
            assert mode == "wal"

    def test_tables_created(self, db):
        """All three tables should exist."""
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "jobs" in tables
        assert "results" in tables
        assert "audit_log" in tables


# ==================== JOB OPERATIONS ====================


class TestJobOperations:
    """Tests for job CRUD operations."""

    def test_create_job(self, db):
        job = db.create_job(
            job_type="invoice_process",
            file_name="invoice.pdf",
            vendor_name="Acme Corp",
        )

        assert job.id is not None
        assert job.type == "invoice_process"
        assert job.status == "PENDING"
        assert job.file_name == "invoice.pdf"
        assert job.vendor_name == "Acme Corp"
        assert job.created_at is not None
        assert job.completed_at is None
        assert job.error is None

    def test_create_job_minimal(self, db):
        """Job with only required fields."""
        job = db.create_job(job_type="contract_ingest")
        assert job.id is not None
        assert job.file_name is None
        assert job.vendor_name is None

    def test_update_job_status(self, db):
        job = db.create_job(job_type="invoice_process")
        updated = db.update_job_status(job.id, "PARSING")

        assert updated.status == "PARSING"
        assert updated.completed_at is None

    def test_update_job_to_complete(self, db):
        job = db.create_job(job_type="invoice_process")
        updated = db.update_job_status(job.id, "COMPLETE")

        assert updated.status == "COMPLETE"
        assert updated.completed_at is not None

    def test_update_job_to_failed(self, db):
        job = db.create_job(job_type="invoice_process")
        updated = db.update_job_status(job.id, "FAILED", error="Ollama timeout")

        assert updated.status == "FAILED"
        assert updated.error == "Ollama timeout"
        assert updated.completed_at is not None

    def test_update_nonexistent_job(self, db):
        result = db.update_job_status(999, "COMPLETE")
        assert result is None

    def test_get_job(self, db):
        created = db.create_job(job_type="po_ingest", file_name="po.pdf")
        fetched = db.get_job(created.id)

        assert fetched is not None
        assert fetched.type == "po_ingest"
        assert fetched.file_name == "po.pdf"

    def test_get_nonexistent_job(self, db):
        assert db.get_job(999) is None

    def test_list_jobs(self, db):
        db.create_job(job_type="contract_ingest")
        db.create_job(job_type="invoice_process")
        db.create_job(job_type="invoice_process")

        all_jobs = db.list_jobs()
        assert len(all_jobs) == 3

    def test_list_jobs_filter_by_type(self, db):
        db.create_job(job_type="contract_ingest")
        db.create_job(job_type="invoice_process")
        db.create_job(job_type="invoice_process")

        invoice_jobs = db.list_jobs(job_type="invoice_process")
        assert len(invoice_jobs) == 2

    def test_list_jobs_filter_by_status(self, db):
        job1 = db.create_job(job_type="invoice_process")
        job2 = db.create_job(job_type="invoice_process")
        db.update_job_status(job1.id, "COMPLETE")

        pending = db.list_jobs(status="PENDING")
        assert len(pending) == 1

    def test_list_jobs_limit(self, db):
        for i in range(10):
            db.create_job(job_type="invoice_process")

        limited = db.list_jobs(limit=3)
        assert len(limited) == 3

    def test_list_jobs_ordered_by_created_desc(self, db):
        job1 = db.create_job(job_type="invoice_process", file_name="first.pdf")
        job2 = db.create_job(job_type="invoice_process", file_name="second.pdf")

        jobs = db.list_jobs()
        # Most recent first
        assert jobs[0].file_name == "second.pdf"
        assert jobs[1].file_name == "first.pdf"


# ==================== RESULT OPERATIONS ====================


class TestResultOperations:
    """Tests for validation result storage."""

    def test_save_result(self, db):
        job = db.create_job(job_type="invoice_process")
        result = db.save_result(
            job_id=job.id,
            invoice_file="invoice.pdf",
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            status="PASS",
            confidence=0.95,
            matches_passed=3,
            total_matches=3,
            details={"test": "data"},
        )

        assert result.id is not None
        assert result.job_id == job.id
        assert result.status == "PASS"
        assert result.confidence == 0.95
        assert result.matches_passed == 3

    def test_result_details_json(self, db):
        job = db.create_job(job_type="invoice_process")
        details = {"issues": [{"rule": "rate", "severity": "critical"}]}
        result = db.save_result(job_id=job.id, details=details)

        assert result.details_json is not None
        parsed = json.loads(result.details_json)
        assert parsed["issues"][0]["rule"] == "rate"

    def test_result_details_property(self, db):
        job = db.create_job(job_type="invoice_process")
        result = db.save_result(
            job_id=job.id,
            details={"key": "value"},
        )

        assert result.details == {"key": "value"}

    def test_result_details_none(self, db):
        job = db.create_job(job_type="invoice_process")
        result = db.save_result(job_id=job.id)

        assert result.details is None

    def test_get_results_by_job(self, db):
        job1 = db.create_job(job_type="invoice_process")
        job2 = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job1.id, invoice_number="INV-001")
        db.save_result(job_id=job1.id, invoice_number="INV-002")
        db.save_result(job_id=job2.id, invoice_number="INV-003")

        job1_results = db.get_results(job_id=job1.id)
        assert len(job1_results) == 2

    def test_get_results_by_vendor(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, vendor_name="Acme", invoice_number="INV-001")
        db.save_result(job_id=job.id, vendor_name="Globex", invoice_number="INV-002")

        acme_results = db.get_results(vendor_name="Acme")
        assert len(acme_results) == 1
        assert acme_results[0].vendor_name == "Acme"

    def test_get_results_by_status(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, status="PASS")
        db.save_result(job_id=job.id, status="FAIL")
        db.save_result(job_id=job.id, status="PASS")

        passed = db.get_results(status="PASS")
        assert len(passed) == 2

    def test_find_duplicate_invoice_matches_vendor_and_number_case_insensitively(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(
            job_id=job.id,
            invoice_file="first.pdf",
            vendor_name=" Acme Corp ",
            invoice_number=" inv-001 ",
            status="PASS",
        )

        duplicate = db.find_duplicate_invoice("acme corp", "INV-001")

        assert duplicate is not None
        assert duplicate["job_id"] == job.id
        assert duplicate["invoice_file"] == "first.pdf"
        assert duplicate["vendor_name"] == " Acme Corp "
        assert duplicate["invoice_number"] == " inv-001 "

    def test_find_duplicate_invoice_does_not_match_different_vendor_or_number(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(
            job_id=job.id,
            vendor_name="Acme Corp",
            invoice_number="INV-001",
        )

        assert db.find_duplicate_invoice("Other Vendor", "INV-001") is None
        assert db.find_duplicate_invoice("Acme Corp", "INV-002") is None

    def test_find_duplicate_invoice_can_exclude_current_job(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(
            job_id=job.id,
            vendor_name="Acme Corp",
            invoice_number="INV-001",
        )

        assert db.find_duplicate_invoice(
            "Acme Corp", "INV-001", exclude_job_id=job.id
        ) is None

    def test_find_duplicate_invoice_ignores_blank_identity_fields(self, db):
        assert db.find_duplicate_invoice("", "INV-001") is None
        assert db.find_duplicate_invoice("Acme Corp", "   ") is None

    def test_job_result_cascade_delete(self, db):
        """Deleting a job should delete its results."""
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, invoice_number="INV-001")
        db.save_result(job_id=job.id, invoice_number="INV-002")

        with db.session() as session:
            job_obj = session.get(Job, job.id)
            session.delete(job_obj)
            session.commit()

        assert db.get_results(job_id=job.id) == []


# ==================== AUDIT LOG ====================


class TestAuditLog:
    """Tests for audit trail."""

    def test_job_creation_audited(self, db):
        db.create_job(job_type="contract_ingest", file_name="contract.pdf")

        logs = db.get_audit_log()
        assert len(logs) >= 1
        assert any(log.action == "job_created" for log in logs)

    def test_status_change_audited(self, db):
        job = db.create_job(job_type="invoice_process")
        db.update_job_status(job.id, "PARSING")

        logs = db.get_audit_log(entity_type="job")
        status_logs = [l for l in logs if l.action == "job_status_changed"]
        assert len(status_logs) == 1

        metadata = json.loads(status_logs[0].extra_data)
        assert metadata["new_status"] == "PARSING"

    def test_result_save_audited(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, invoice_number="INV-001", status="PASS")

        logs = db.get_audit_log(entity_type="result")
        assert len(logs) >= 1
        assert any(log.action == "result_saved" for log in logs)

    def test_audit_log_filter_by_entity_type(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, status="PASS")

        job_logs = db.get_audit_log(entity_type="job")
        result_logs = db.get_audit_log(entity_type="result")

        # Job logs: job_created + job event for the create
        # Result logs: result_saved
        assert all(log.entity_type == "job" for log in job_logs)
        assert all(log.entity_type == "result" for log in result_logs)

    def test_audit_log_limit(self, db):
        for _ in range(10):
            db.create_job(job_type="invoice_process")

        limited = db.get_audit_log(limit=3)
        assert len(limited) == 3


# ==================== STATS ====================


class TestStats:
    """Tests for aggregate statistics."""

    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["total_jobs"] == 0
        assert stats["total_results"] == 0
        assert stats["pass_rate"] == 0.0

    def test_stats_with_data(self, db):
        # Create some jobs
        job1 = db.create_job(job_type="invoice_process")
        job2 = db.create_job(job_type="invoice_process")
        job3 = db.create_job(job_type="invoice_process")

        db.update_job_status(job1.id, "COMPLETE")
        db.update_job_status(job2.id, "COMPLETE")
        db.update_job_status(job3.id, "FAILED", error="timeout")

        # Create some results
        db.save_result(job_id=job1.id, status="PASS")
        db.save_result(job_id=job1.id, status="PASS")
        db.save_result(job_id=job2.id, status="FAIL")
        db.save_result(job_id=job2.id, status="REVIEW")

        stats = db.get_stats()
        assert stats["total_jobs"] == 3
        assert stats["completed_jobs"] == 2
        assert stats["failed_jobs"] == 1
        assert stats["pending_jobs"] == 0
        assert stats["total_results"] == 4
        assert stats["pass_count"] == 2
        assert stats["fail_count"] == 1
        assert stats["review_count"] == 1
        assert stats["pass_rate"] == 0.5

    def test_pass_rate_calculation(self, db):
        job = db.create_job(job_type="invoice_process")
        db.save_result(job_id=job.id, status="PASS")
        db.save_result(job_id=job.id, status="PASS")
        db.save_result(job_id=job.id, status="PASS")
        db.save_result(job_id=job.id, status="FAIL")

        stats = db.get_stats()
        assert stats["pass_rate"] == 0.75


# ==================== CONCURRENCY ====================


class TestConcurrency:
    """Tests for concurrent access patterns."""

    def test_concurrent_duplicate_identity_is_serialized(self, db):
        """Only the first concurrent save may be stored without REVIEW."""
        jobs = [
            db.create_job(job_type="invoice_process"),
            db.create_job(job_type="invoice_process"),
        ]
        barrier = Barrier(2)

        def save(job):
            barrier.wait(timeout=5)
            return db.save_result_with_duplicate_check(
                job_id=job.id,
                invoice_file=f"{job.id}.pdf",
                vendor_name="Acme Corp",
                invoice_number="INV-RACE-001",
                status="PASS",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(save, jobs))

        statuses = sorted(result.status for result, _ in results)
        assert statuses == ["PASS", "REVIEW"]
        assert sum(duplicate is not None for _, duplicate in results) == 1
        assert len(db.get_results()) == 2

    def test_multiple_sessions(self, db):
        """Multiple sessions should work without conflicts (WAL mode)."""
        job = db.create_job(job_type="invoice_process")

        # Read from one session while writing from another
        with db.session() as s1:
            j1 = s1.get(Job, job.id)
            assert j1.status == "PENDING"

        db.update_job_status(job.id, "COMPLETE")

        with db.session() as s2:
            j2 = s2.get(Job, job.id)
            assert j2.status == "COMPLETE"



# ==================== VENDOR BANK-DETAIL MONITORING ====================


class TestVendorBankDetailMonitoring:
    def test_first_bank_profile_is_registered_without_alert(self, db):
        result = db.record_vendor_bank_details(
            "Acme Corp", "Example Bank", "1234", "hmac-v1:initial"
        )
        assert result["status"] == "REGISTERED"
        assert result["review_required"] is False

    def test_changed_bank_fingerprint_creates_review_alert(self, db):
        db.record_vendor_bank_details(
            "Acme Corp", "Example Bank", "1234", "hmac-v1:initial"
        )
        result = db.record_vendor_bank_details(
            " ACME CORP ", "New Bank", "9876", "hmac-v1:changed"
        )
        assert result["status"] == "REVIEW"
        assert result["review_required"] is True
        assert result["previous"] == {
            "bank_name": "Example Bank", "account_last4": "1234"
        }
        assert result["current"] == {
            "bank_name": "New Bank", "account_last4": "9876"
        }
        logs = db.get_audit_log(entity_type="vendor")
        alert = next(log for log in logs if log.action == "vendor_bank_details_changed")
        metadata = json.loads(alert.extra_data)
        assert metadata["review_required"] is True
        assert metadata["payment_authorized"] is False

    def test_same_fingerprint_does_not_create_alert(self, db):
        db.record_vendor_bank_details(
            "Acme Corp", "Example Bank", "1234", "hmac-v1:initial"
        )
        result = db.record_vendor_bank_details(
            "Acme Corp", "Example Bank", "1234", "hmac-v1:initial"
        )
        assert result["status"] == "UNCHANGED"
        assert result["review_required"] is False

    def test_invalid_bank_identity_is_rejected(self, db):
        with pytest.raises(ValueError):
            db.record_vendor_bank_details(
                "Acme Corp", "Example Bank", "12345", "hmac-v1:initial"
            )


class TestSuspiciousInvoicePatterns:
    def test_finds_repeated_same_vendor_amount_distinct_invoice_numbers(self, db):
        for number in ("INV-A", "INV-B", "INV-C"):
            job = db.create_job(job_type="invoice_process")
            db.save_result(
                job_id=job.id,
                vendor_name="Acme Corp",
                invoice_number=number,
                status="PASS",
                details={"total_amount": 1250.00, "currency": "USD"},
            )
        findings = db.find_suspicious_invoice_patterns()
        assert len(findings) == 1
        assert findings[0]["rule"] == "repeated_same_vendor_amount"
        assert findings[0]["occurrences"] == 3
        assert {item["invoice_number"] for item in findings[0]["evidence"]} == {"INV-A", "INV-B", "INV-C"}
        assert findings[0]["status"] == "REVIEW"

    def test_does_not_flag_different_currency_or_amount(self, db):
        for number, amount, currency in (
            ("INV-A", 100, "USD"),
            ("INV-B", 100, "EUR"),
            ("INV-C", 101, "USD"),
        ):
            job = db.create_job(job_type="invoice_process")
            db.save_result(job_id=job.id, vendor_name="Acme", invoice_number=number,
                           details={"total_amount": amount, "currency": currency})
        assert db.find_suspicious_invoice_patterns() == []

    def test_same_invoice_identity_does_not_inflate_pattern_count(self, db):
        job = db.create_job(job_type="invoice_process")
        for _ in range(3):
            db.save_result(job_id=job.id, vendor_name="Acme", invoice_number="INV-1",
                           details={"total_amount": 100, "currency": "USD"})
        assert db.find_suspicious_invoice_patterns() == []

    def test_validates_detection_parameters(self, db):
        with pytest.raises(ValueError):
            db.find_suspicious_invoice_patterns(lookback_days=0)
        with pytest.raises(ValueError):
            db.find_suspicious_invoice_patterns(min_occurrences=1)



class TestPaymentRecords:
    def test_records_payment_for_review_and_audits_without_raw_bank_data(self, db):
        from datetime import date

        job = db.create_job(job_type="invoice_process")
        invoice = db.save_result(
            job_id=job.id,
            vendor_name="Acme Corp",
            invoice_number="INV-200",
            status="PASS",
        )
        payment = db.create_payment_record(
            vendor_name="Acme Corp",
            invoice_result_id=invoice.id,
            payment_reference="BANK-REF-001",
            amount="1250.50",
            currency="usd",
            payment_date=date(2026, 11, 8),
            bank_account_fingerprint="opaque-keyed-hmac",
            source="IMPORT",
        )

        assert payment["vendor_name"] == "Acme Corp"
        assert payment["invoice_number"] == "INV-200"
        assert payment["invoice_result_id"] == invoice.id
        assert payment["amount"] == "1250.50"
        assert payment["currency"] == "USD"
        assert payment["payment_date"] == "2026-11-08"
        assert payment["source"] == "IMPORT"
        assert payment["status"] == "PENDING_REVIEW"
        assert "bank_account_fingerprint" not in payment

        audit = db.get_audit_log(entity_type="payment", limit=5)
        assert any(
            entry.action == "payment_recorded"
            and entry.audit_metadata["payment_authorized"] is False
            for entry in audit
        )

    @pytest.mark.parametrize("amount", ["0", "-1", "not-a-number", "12.345"])
    def test_rejects_invalid_payment_amount(self, db, amount):
        with pytest.raises(ValueError):
            db.create_payment_record(
                vendor_name="Acme Corp", amount=amount, currency="USD"
            )

    def test_rejects_invalid_currency_and_source(self, db):
        with pytest.raises(ValueError):
            db.create_payment_record(
                vendor_name="Acme Corp", amount="10.00", currency="US"
            )
        with pytest.raises(ValueError):
            db.create_payment_record(
                vendor_name="Acme Corp", amount="10.00", currency="USD",
                source="EXECUTED",
            )

    def test_rejects_missing_or_mismatched_invoice_link(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.create_payment_record(
                vendor_name="Acme Corp", amount="10.00", currency="USD",
                invoice_result_id=999,
            )

        job = db.create_job(job_type="invoice_process")
        invoice = db.save_result(
            job_id=job.id, vendor_name="Different Vendor",
            invoice_number="INV-1", status="PASS",
        )
        with pytest.raises(ValueError, match="does not match"):
            db.create_payment_record(
                vendor_name="Acme Corp", amount="10.00", currency="USD",
                invoice_result_id=invoice.id,
            )

    def test_lists_payment_records_with_filters(self, db):
        db.create_payment_record(
            vendor_name="Acme Corp", amount="10.00", currency="USD"
        )
        db.create_payment_record(
            vendor_name="Other Corp", amount="20.00", currency="GBP"
        )

        rows = db.list_payment_records(vendor_name="acme corp")
        assert len(rows) == 1
        assert rows[0]["vendor_name"] == "Acme Corp"
        assert db.list_payment_records(status="PENDING_REVIEW")[0]["status"] == "PENDING_REVIEW"

    def test_rejects_unsupported_payment_status_filter(self, db):
        with pytest.raises(ValueError):
            db.list_payment_records(status="EXECUTED")



class TestDuplicatePaymentCandidates:
    def test_flags_repeated_payment_reference_for_same_vendor_and_currency(self, db):
        for _ in range(2):
            db.create_payment_record(
                vendor_name="Acme Corp", payment_reference="BANK-REF-9",
                amount="100.00", currency="USD",
            )
        findings = db.find_duplicate_payment_candidates()
        reference_findings = [
            item for item in findings if item["rule"] == "duplicate_payment_reference"
        ]
        assert len(reference_findings) == 1
        assert reference_findings[0]["status"] == "REVIEW"
        assert reference_findings[0]["occurrences"] == 2
        assert len(reference_findings[0]["payment_ids"]) == 2

    def test_flags_same_invoice_amount_and_currency_recorded_twice(self, db):
        for _ in range(2):
            db.create_payment_record(
                vendor_name="Acme Corp", invoice_number="INV-55",
                amount="250.00", currency="USD",
            )
        findings = db.find_duplicate_payment_candidates()
        invoice_findings = [
            item for item in findings if item["rule"] == "repeated_invoice_payment"
        ]
        assert len(invoice_findings) == 1
        assert invoice_findings[0]["invoice_number"] == "INV-55"
        assert invoice_findings[0]["amount"] == "250.00"
        assert invoice_findings[0]["occurrences"] == 2

    def test_does_not_group_different_vendor_currency_or_invoice_amount(self, db):
        db.create_payment_record(
            vendor_name="Acme Corp", invoice_number="INV-1",
            amount="100.00", currency="USD", payment_reference="REF-1",
        )
        db.create_payment_record(
            vendor_name="Other Corp", invoice_number="INV-1",
            amount="100.00", currency="USD", payment_reference="REF-1",
        )
        db.create_payment_record(
            vendor_name="Acme Corp", invoice_number="INV-1",
            amount="100.00", currency="EUR", payment_reference="REF-1",
        )
        db.create_payment_record(
            vendor_name="Acme Corp", invoice_number="INV-1",
            amount="101.00", currency="USD", payment_reference="REF-1",
        )
        # Same reference is only compared within vendor and currency; invoice
        # amount rule additionally requires the same amount.
        findings = db.find_duplicate_payment_candidates()
        assert not any(
            item["rule"] == "repeated_invoice_payment" for item in findings
        )
        assert not any(
            item["rule"] == "duplicate_payment_reference"
            and item["vendor_name"] == "Acme Corp"
            and item["currency"] == "USD"
            for item in findings
        )

    def test_vendor_filter_and_limit_validation(self, db):
        for vendor in ("Acme Corp", "Other Corp"):
            for _ in range(2):
                db.create_payment_record(
                    vendor_name=vendor, payment_reference="SAME-REF",
                    amount="10.00", currency="USD",
                )
        findings = db.find_duplicate_payment_candidates(vendor_name="acme corp")
        assert all(item["vendor_name"] == "Acme Corp" for item in findings)
        with pytest.raises(ValueError):
            db.find_duplicate_payment_candidates(limit=0)


class TestResultEvidence:
    def test_get_result_evidence_returns_stored_details(self, db):
        job = db.create_job(job_type="invoice_process")
        result = db.save_result(
            job_id=job.id,
            vendor_name="Acme Corp",
            invoice_number="INV-100",
            status="REVIEW",
            details={
                "risk_explanation": {
                    "summary": "One warning requires review.",
                    "severity_counts": {"critical": 0, "error": 0, "warning": 1, "info": 0},
                },
                "issue_evidence": [{"rule": "price_variance", "severity": "warning"}],
            },
        )
        evidence = db.get_result_evidence(result.id)
        assert evidence["result_id"] == result.id
        assert evidence["details"]["risk_explanation"]["severity_counts"]["warning"] == 1
        assert evidence["details"]["issue_evidence"][0]["rule"] == "price_variance"

    def test_get_result_evidence_returns_none_for_unknown_id(self, db):
        assert db.get_result_evidence(999999) is None

    def test_get_result_evidence_rejects_non_positive_id(self, db):
        assert db.get_result_evidence(0) is None

