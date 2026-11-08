"""
SQLite Metadata Store for DocuMatch Architect.

Provides persistent storage for jobs, validation results, and audit logs.
Uses SQLAlchemy ORM with SQLite backend and WAL mode for concurrent access.

Tables:
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
    │    jobs       │     │    results        │     │  audit_log   │
    ├──────────────┤     ├──────────────────┤     ├──────────────┤
    │ id (PK)      │──┐  │ id (PK)          │     │ id (PK)      │
    │ type         │  └─▶│ job_id (FK)       │     │ action       │
    │ status       │     │ invoice_file      │     │ entity_type  │
    │ file_name    │     │ vendor_name       │     │ entity_id    │
    │ vendor_name  │     │ invoice_number    │     │ timestamp    │
    │ created_at   │     │ status            │     │ metadata_json│
    │ completed_at │     │ confidence        │     └──────────────┘
    │ error        │     │ details_json      │
    └──────────────┘     │ created_at        │
                         └──────────────────┘

Job State Machine:
    PENDING ──▶ PARSING ──▶ EXTRACTING ──▶ MATCHING ──▶ COMPLETE
       │           │            │              │
       └───────────┴────────────┴──────────────┴──▶ FAILED
"""

import json
import logging
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    event,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

logger = logging.getLogger(__name__)

# Valid job statuses
JOB_STATUSES = ("PENDING", "PARSING", "EXTRACTING", "MATCHING", "COMPLETE", "FAILED")

# Valid job types
JOB_TYPES = ("contract_ingest", "po_ingest", "invoice_process", "batch_process")


# ==================== ORM MODELS ====================


class Base(DeclarativeBase):
    pass


class Job(Base):
    """Tracks processing jobs (contract ingestion, PO processing, invoice validation)."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")
    file_name = Column(String(255), nullable=True)
    vendor_name = Column(String(255), nullable=True)
    parent_job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    # Relationship to results
    results = relationship("Result", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Job(id={self.id}, type={self.type}, status={self.status})>"


class Result(Base):
    """Stores validation results for processed invoices."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    invoice_file = Column(String(255), nullable=True)
    vendor_name = Column(String(255), nullable=True)
    invoice_number = Column(String(100), nullable=True)
    status = Column(String(20), nullable=True)  # PASS, FAIL, REVIEW
    confidence = Column(Float, nullable=True)
    matches_passed = Column(Integer, nullable=True)
    total_matches = Column(Integer, nullable=True)
    details_json = Column(Text, nullable=True)  # Full result as JSON
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationship to job
    job = relationship("Job", back_populates="results")

    @property
    def details(self) -> Optional[dict]:
        """Parse details_json into a dict."""
        if self.details_json:
            return json.loads(self.details_json)
        return None

    @details.setter
    def details(self, value: dict):
        """Serialize dict to details_json."""
        self.details_json = json.dumps(value) if value else None

    def __repr__(self):
        return f"<Result(id={self.id}, invoice={self.invoice_number}, status={self.status})>"


class VendorBankProfile(Base):
    """Current vendor bank identity; never stores a full account number."""

    __tablename__ = "vendor_bank_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vendor_key = Column(String(255), nullable=False, unique=True, index=True)
    vendor_name = Column(String(255), nullable=False)
    bank_name = Column(String(255), nullable=False)
    account_last4 = Column(String(4), nullable=False)
    account_fingerprint = Column(String(128), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class Payment(Base):
    """A recorded payment instruction or payment event; never stores raw bank details."""

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vendor_name = Column(String(255), nullable=False, index=True)
    invoice_number = Column(String(100), nullable=True, index=True)
    invoice_result_id = Column(Integer, ForeignKey("results.id"), nullable=True, index=True)
    payment_reference = Column(String(150), nullable=True, index=True)
    amount = Column(Numeric(18, 2), nullable=False)
    currency = Column(String(3), nullable=False)
    payment_date = Column(DateTime, nullable=True)
    bank_account_fingerprint = Column(String(128), nullable=True)
    source = Column(String(20), nullable=False, default="MANUAL")
    status = Column(String(30), nullable=False, default="PENDING_REVIEW")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    """Audit trail for significant system actions."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)  # contract, po, invoice, job
    entity_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    extra_data = Column("metadata_json", Text, nullable=True)

    @property
    def audit_metadata(self) -> Optional[dict]:
        """Parse extra_data JSON into a dict."""
        if self.extra_data:
            return json.loads(self.extra_data)
        return None

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action})>"


# ==================== DATABASE MANAGER ====================


class Database:
    """
    Manages SQLite database connection and session lifecycle.

    Uses WAL mode for concurrent read/write access from
    both Streamlit and FastAPI processes.

    Usage:
        db = Database("./data/documatch.db")
        with db.session() as session:
            job = Job(type="invoice_process", status="PENDING")
            session.add(job)
            session.commit()
    """

    def __init__(self, db_path: str = "./data/documatch.db"):
        self.db_path = db_path

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite-specific settings
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Enable WAL mode for concurrent access
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        # Create session factory (expire_on_commit=False so returned objects
        # remain usable after session close)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Upgrade databases created before batch parent-child links existed.
        with self.engine.begin() as connection:
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(jobs)").fetchall()}
            if "parent_job_id" not in columns:
                connection.exec_driver_sql("ALTER TABLE jobs ADD COLUMN parent_job_id INTEGER REFERENCES jobs(id)")
                connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_jobs_parent_job_id ON jobs(parent_job_id)")

        logger.info(f"Database initialized at {db_path} (WAL mode)")

    def session(self) -> Session:
        """Create a new database session. Use with context manager."""
        return self.SessionLocal()

    # ==================== JOB OPERATIONS ====================

    def create_job(
        self,
        job_type: str,
        file_name: Optional[str] = None,
        vendor_name: Optional[str] = None,
        parent_job_id: Optional[int] = None,
    ) -> Job:
        """Create a new processing job, optionally linked to a batch parent."""
        with self.session() as session:
            job = Job(
                type=job_type,
                status="PENDING",
                file_name=file_name,
                vendor_name=vendor_name,
                parent_job_id=parent_job_id,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id

            # Audit log
            self._audit(session, "job_created", "job", str(job_id), {
                "type": job_type, "file_name": file_name,
                "parent_job_id": parent_job_id,
            })
            session.commit()

            logger.info(f"Created job {job_id}: type={job_type}")
            return job

    def update_job_status(
        self,
        job_id: int,
        status: str,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update a job's status. Sets completed_at for terminal states."""
        with self.session() as session:
            job = session.get(Job, job_id)
            if not job:
                return None

            job.status = status
            if error:
                job.error = error
            if status in ("COMPLETE", "FAILED"):
                job.completed_at = datetime.now(timezone.utc)

            self._audit(session, "job_status_changed", "job", str(job_id), {
                "new_status": status, "error": error,
            })
            session.commit()
            session.refresh(job)

            return job

    def get_job(self, job_id: int) -> Optional[Job]:
        """Get a job by ID."""
        with self.session() as session:
            return session.get(Job, job_id)

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs with optional filters."""
        with self.session() as session:
            query = session.query(Job)
            if job_type:
                query = query.filter(Job.type == job_type)
            if status:
                query = query.filter(Job.status == status)
            return query.order_by(Job.created_at.desc()).limit(limit).all()

    # ==================== RESULT OPERATIONS ====================

    def save_result(
        self,
        job_id: int,
        invoice_file: Optional[str] = None,
        vendor_name: Optional[str] = None,
        invoice_number: Optional[str] = None,
        status: Optional[str] = None,
        confidence: Optional[float] = None,
        matches_passed: Optional[int] = None,
        total_matches: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> Result:
        """Save a validation result linked to a job."""
        with self.session() as session:
            result = Result(
                job_id=job_id,
                invoice_file=invoice_file,
                vendor_name=vendor_name,
                invoice_number=invoice_number,
                status=status,
                confidence=confidence,
                matches_passed=matches_passed,
                total_matches=total_matches,
                details_json=json.dumps(details) if details else None,
            )
            session.add(result)

            self._audit(session, "result_saved", "result", invoice_number, {
                "job_id": job_id, "status": status,
            })
            session.commit()
            session.refresh(result)

            return result

    def get_results(
        self,
        job_id: Optional[int] = None,
        vendor_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Result]:
        """Query results with optional filters."""
        with self.session() as session:
            query = session.query(Result)
            if job_id:
                query = query.filter(Result.job_id == job_id)
            if vendor_name:
                query = query.filter(Result.vendor_name == vendor_name)
            if status:
                query = query.filter(Result.status == status)
            return query.order_by(Result.created_at.desc()).limit(limit).all()

    def save_result_with_duplicate_check(
        self,
        job_id: int,
        invoice_file: Optional[str] = None,
        vendor_name: Optional[str] = None,
        invoice_number: Optional[str] = None,
        status: Optional[str] = None,
        confidence: Optional[float] = None,
        matches_passed: Optional[int] = None,
        total_matches: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> tuple[Result, Optional[dict]]:
        """
        Atomically check invoice identity and persist its result.

        BEGIN IMMEDIATE obtains SQLite's reserved write lock before the
        duplicate lookup. Concurrent writers therefore cannot both observe
        the same identity as absent and then save it as a first occurrence.
        """
        vendor_key = (vendor_name or "").strip()
        invoice_key = (invoice_number or "").strip()

        with self.session() as session:
            # Acquire the SQLite write reservation before reading identity.
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")

            duplicate = None
            if vendor_key and invoice_key:
                query = session.query(Result).filter(
                    Result.vendor_name.isnot(None),
                    Result.invoice_number.isnot(None),
                    func.upper(func.trim(Result.vendor_name)) == vendor_key.upper(),
                    func.upper(func.trim(Result.invoice_number)) == invoice_key.upper(),
                    Result.job_id != job_id,
                )
                previous = query.order_by(
                    Result.created_at.desc(), Result.id.desc()
                ).first()
                if previous is not None:
                    duplicate = {
                        "result_id": previous.id,
                        "job_id": previous.job_id,
                        "invoice_file": previous.invoice_file,
                        "vendor_name": previous.vendor_name,
                        "invoice_number": previous.invoice_number,
                        "status": previous.status,
                        "created_at": (
                            previous.created_at.isoformat()
                            if previous.created_at else None
                        ),
                    }

            stored_status = "REVIEW" if duplicate else status
            persisted_details = dict(details or {})
            persisted_details["duplicate_invoice_candidate"] = duplicate

            # A duplicate identity changes the persisted decision to REVIEW.
            # Keep the explanation consistent with that final stored status.
            if duplicate:
                risk_explanation = dict(
                    persisted_details.get("risk_explanation") or {}
                )
                risk_explanation["status"] = "REVIEW"
                risk_explanation["human_review_required"] = True
                risk_explanation["duplicate_invoice_candidate"] = True
                risk_explanation["summary"] = (
                    str(risk_explanation.get("summary", "")).strip()
                    + " A previously processed vendor/invoice identity was found; "
                    "human verification is required."
                ).strip()
                persisted_details["risk_explanation"] = risk_explanation
            result = Result(
                job_id=job_id,
                invoice_file=invoice_file,
                vendor_name=vendor_name,
                invoice_number=invoice_number,
                status=stored_status,
                confidence=confidence,
                matches_passed=matches_passed,
                total_matches=total_matches,
                details_json=json.dumps(persisted_details) if persisted_details else None,
            )
            session.add(result)
            self._audit(session, "result_saved", "result", invoice_number, {
                "job_id": job_id, "status": stored_status,
                "duplicate_candidate": duplicate is not None,
            })
            session.commit()
            session.refresh(result)
            return result, duplicate

    def find_duplicate_invoice(
        self,
        vendor_name: str,
        invoice_number: str,
        exclude_job_id: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Find a previously processed invoice with the same vendor and invoice number.

        Matching is case-insensitive and ignores leading/trailing whitespace.
        Amount and currency are deliberately not used as identity fields: a
        repeated invoice reference remains a duplicate candidate even if its
        extracted amount differs. The caller must route the candidate for
        human review rather than treating it as proof of fraud.
        """
        vendor_key = (vendor_name or "").strip()
        invoice_key = (invoice_number or "").strip()
        if not vendor_key or not invoice_key:
            return None

        with self.session() as session:
            query = session.query(Result).filter(
                Result.vendor_name.isnot(None),
                Result.invoice_number.isnot(None),
                func.upper(func.trim(Result.vendor_name)) == vendor_key.upper(),
                func.upper(func.trim(Result.invoice_number)) == invoice_key.upper(),
            )
            if exclude_job_id is not None:
                query = query.filter(Result.job_id != exclude_job_id)

            result = query.order_by(Result.created_at.desc(), Result.id.desc()).first()
            if result is None:
                return None
            return {
                "result_id": result.id,
                "job_id": result.job_id,
                "invoice_file": result.invoice_file,
                "vendor_name": result.vendor_name,
                "invoice_number": result.invoice_number,
                "status": result.status,
                "created_at": result.created_at.isoformat() if result.created_at else None,
            }

    def get_result_evidence(self, result_id: int) -> Optional[dict]:
        """Return stored validation evidence for one result without exposing file contents."""
        if result_id <= 0:
            return None
        with self.session() as session:
            result = session.get(Result, result_id)
            if result is None:
                return None
            return {
                "result_id": result.id,
                "job_id": result.job_id,
                "vendor_name": result.vendor_name,
                "invoice_number": result.invoice_number,
                "status": result.status,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "details": result.details or {},
            }

    def find_suspicious_invoice_patterns(
        self,
        lookback_days: int = 7,
        min_occurrences: int = 3,
        vendor_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Find repeated same-value invoices for a vendor within a processing window.

        This is a review indicator, not a fraud conclusion. The time window is
        based on result processing timestamps because invoice dates are not
        persisted as first-class result columns.
        """
        if not 1 <= lookback_days <= 365:
            raise ValueError("lookback_days must be between 1 and 365.")
        if not 2 <= min_occurrences <= 20:
            raise ValueError("min_occurrences must be between 2 and 20.")

        from datetime import timedelta
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=lookback_days)
        with self.session() as session:
            query = session.query(Result).filter(
                Result.created_at >= cutoff,
                Result.vendor_name.isnot(None),
                Result.invoice_number.isnot(None),
            )
            if vendor_name and vendor_name.strip():
                query = query.filter(
                    func.upper(func.trim(Result.vendor_name)) == vendor_name.strip().upper()
                )
            rows = query.order_by(Result.created_at.asc(), Result.id.asc()).all()

        groups = {}
        for row in rows:
            details = row.details or {}
            amount = details.get("total_amount")
            currency = details.get("currency")
            if amount is None or not currency:
                continue
            try:
                amount_value = round(float(amount), 2)
            except (TypeError, ValueError):
                continue
            if amount_value <= 0:
                continue
            vendor_key = (row.vendor_name or "").strip().casefold()
            invoice_key = (row.invoice_number or "").strip().casefold()
            if not vendor_key or not invoice_key:
                continue
            key = (vendor_key, amount_value, str(currency).strip().upper())
            groups.setdefault(key, {})
            # Repeated processing of the same invoice identity is already
            # handled by duplicate detection; only distinct references count here.
            groups[key].setdefault(invoice_key, row)

        findings = []
        for (_, amount, currency), invoices in groups.items():
            if len(invoices) < min_occurrences:
                continue
            records = sorted(invoices.values(), key=lambda item: item.created_at)
            findings.append({
                "rule": "repeated_same_vendor_amount",
                "status": "REVIEW",
                "vendor_name": records[0].vendor_name,
                "amount": amount,
                "currency": currency,
                "occurrences": len(records),
                "lookback_days": lookback_days,
                "first_processed_at": records[0].created_at.isoformat() if records[0].created_at else None,
                "last_processed_at": records[-1].created_at.isoformat() if records[-1].created_at else None,
                "evidence": [
                    {
                        "result_id": item.id,
                        "job_id": item.job_id,
                        "invoice_number": item.invoice_number,
                        "processed_at": item.created_at.isoformat() if item.created_at else None,
                        "validation_status": item.status,
                    }
                    for item in records
                ],
                "explanation": (
                    "Multiple distinct invoice references from this vendor have "
                    "the same amount and currency within the selected processing window. "
                    "This may be legitimate recurring billing and requires human review."
                ),
            })
        return sorted(findings, key=lambda item: (-item["occurrences"], item["vendor_name"].casefold()))

    # ==================== VENDOR BANK-DETAIL MONITORING ====================

    def record_vendor_bank_details(
        self,
        vendor_name: str,
        bank_name: str,
        account_last4: str,
        account_fingerprint: str,
    ) -> dict:
        """
        Register or compare a vendor bank identity.

        The caller must provide a keyed HMAC fingerprint of normalized bank
        details. Raw account/routing numbers must never be passed or stored.
        A changed fingerprint updates the profile and creates an auditable
        REVIEW alert; it does not authorize a payment.
        """
        name = (vendor_name or "").strip()
        bank = (bank_name or "").strip()
        last4 = (account_last4 or "").strip()
        fingerprint = (account_fingerprint or "").strip()
        if not name or not bank or len(last4) != 4 or not last4.isdigit() or not fingerprint:
            raise ValueError("Vendor, bank name, four-digit account suffix, and keyed fingerprint are required.")

        vendor_key = name.casefold()
        with self.session() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            profile = session.query(VendorBankProfile).filter(
                VendorBankProfile.vendor_key == vendor_key
            ).first()
            if profile is None:
                profile = VendorBankProfile(
                    vendor_key=vendor_key,
                    vendor_name=name,
                    bank_name=bank,
                    account_last4=last4,
                    account_fingerprint=fingerprint,
                )
                session.add(profile)
                self._audit(session, "vendor_bank_details_registered", "vendor", vendor_key, {
                    "bank_name": bank, "account_last4": last4,
                })
                session.commit()
                return {"status": "REGISTERED", "review_required": False, "vendor_name": name}

            changed = profile.account_fingerprint != fingerprint
            previous = {
                "bank_name": profile.bank_name,
                "account_last4": profile.account_last4,
            }
            if changed:
                profile.bank_name = bank
                profile.account_last4 = last4
                profile.account_fingerprint = fingerprint
                profile.updated_at = datetime.now(timezone.utc)
                self._audit(session, "vendor_bank_details_changed", "vendor", vendor_key, {
                    "previous": previous,
                    "current": {"bank_name": bank, "account_last4": last4},
                    "review_required": True,
                    "payment_authorized": False,
                })
            session.commit()
            return {
                "status": "REVIEW" if changed else "UNCHANGED",
                "review_required": changed,
                "vendor_name": name,
                "previous": previous if changed else None,
                "current": {"bank_name": bank, "account_last4": last4},
            }

    # ==================== PAYMENT RECORDS ====================

    def create_payment_record(
        self,
        vendor_name: str,
        amount,
        currency: str,
        invoice_number: Optional[str] = None,
        invoice_result_id: Optional[int] = None,
        payment_reference: Optional[str] = None,
        payment_date: Optional[date | datetime] = None,
        bank_account_fingerprint: Optional[str] = None,
        source: str = "MANUAL",
    ) -> dict:
        """
        Record payment data for review. This does not execute or authorize payment.

        Raw bank account details are never accepted; callers may supply only an
        opaque keyed fingerprint generated outside this database.
        """
        name = (vendor_name or "").strip()
        currency_code = (currency or "").strip().upper()
        source_value = (source or "").strip().upper()
        invoice_ref = (invoice_number or "").strip() or None
        payment_ref = (payment_reference or "").strip() or None
        fingerprint = (bank_account_fingerprint or "").strip() or None
        try:
            amount_value = Decimal(str(amount))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Payment amount must be a valid positive number.")
        if not amount_value.is_finite() or amount_value <= 0:
            raise ValueError("Payment amount must be a valid positive number.")
        if amount_value.as_tuple().exponent < -2:
            raise ValueError("Payment amount may have no more than two decimal places.")
        if not name:
            raise ValueError("Vendor name is required.")
        if len(currency_code) != 3 or not currency_code.isascii() or not currency_code.isalpha():
            raise ValueError("Currency must be a three-letter ISO-style code.")
        if source_value not in ("MANUAL", "IMPORT"):
            raise ValueError("Payment source must be MANUAL or IMPORT.")
        if payment_ref and len(payment_ref) > 150:
            raise ValueError("Payment reference must be 150 characters or fewer.")
        if fingerprint and len(fingerprint) > 128:
            raise ValueError("Bank account fingerprint must be 128 characters or fewer.")

        if isinstance(payment_date, datetime):
            payment_datetime = payment_date
        elif isinstance(payment_date, date):
            payment_datetime = datetime.combine(payment_date, time.min, tzinfo=timezone.utc)
        else:
            payment_datetime = None

        with self.session() as session:
            linked_result = None
            if invoice_result_id is not None:
                if invoice_result_id <= 0:
                    raise ValueError("invoice_result_id must be a positive integer.")
                linked_result = session.get(Result, invoice_result_id)
                if linked_result is None:
                    raise ValueError("Linked invoice result was not found.")
                if (linked_result.vendor_name or "").strip().casefold() != name.casefold():
                    raise ValueError("Payment vendor does not match the linked invoice vendor.")
                if invoice_ref and (linked_result.invoice_number or "").strip().casefold() != invoice_ref.casefold():
                    raise ValueError("Payment invoice number does not match the linked invoice result.")
                if not invoice_ref:
                    invoice_ref = linked_result.invoice_number

            payment = Payment(
                vendor_name=name,
                invoice_number=invoice_ref,
                invoice_result_id=invoice_result_id,
                payment_reference=payment_ref,
                amount=amount_value,
                currency=currency_code,
                payment_date=payment_datetime,
                bank_account_fingerprint=fingerprint,
                source=source_value,
                status="PENDING_REVIEW",
            )
            session.add(payment)
            session.flush()
            self._audit(session, "payment_recorded", "payment", str(payment.id), {
                "vendor_name": name,
                "invoice_number": invoice_ref,
                "invoice_result_id": invoice_result_id,
                "amount": format(amount_value, ".2f"),
                "currency": currency_code,
                "source": source_value,
                "status": "PENDING_REVIEW",
                "bank_fingerprint_supplied": fingerprint is not None,
                "payment_authorized": False,
            })
            session.commit()
            session.refresh(payment)
            return self._payment_dict(payment)

    @staticmethod
    def _payment_dict(payment: Payment) -> dict:
        return {
            "id": payment.id,
            "vendor_name": payment.vendor_name,
            "invoice_number": payment.invoice_number,
            "invoice_result_id": payment.invoice_result_id,
            "payment_reference": payment.payment_reference,
            "amount": format(payment.amount, ".2f"),
            "currency": payment.currency,
            "payment_date": payment.payment_date.date().isoformat() if payment.payment_date else None,
            "source": payment.source,
            "status": payment.status,
            "created_at": payment.created_at.isoformat() if payment.created_at else None,
        }

    def list_payment_records(
        self,
        vendor_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List recorded payment data; does not imply payment execution."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")
        with self.session() as session:
            query = session.query(Payment)
            if vendor_name and vendor_name.strip():
                query = query.filter(
                    func.upper(func.trim(Payment.vendor_name)) == vendor_name.strip().upper()
                )
            if status:
                status_value = status.strip().upper()
                if status_value != "PENDING_REVIEW":
                    raise ValueError("Only PENDING_REVIEW payment records are currently supported.")
                query = query.filter(Payment.status == status_value)
            payments = query.order_by(Payment.created_at.desc(), Payment.id.desc()).limit(limit).all()
            return [self._payment_dict(payment) for payment in payments]

    # ==================== AUDIT OPERATIONS ====================

    def _audit(
        self,
        session: Session,
        action: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Add an audit log entry (called within an existing session)."""
        log = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            extra_data=json.dumps(metadata) if metadata else None,
        )
        session.add(log)

    def get_audit_log(
        self,
        entity_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Query audit log entries."""
        with self.session() as session:
            query = session.query(AuditLog)
            if entity_type:
                query = query.filter(AuditLog.entity_type == entity_type)
            return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()

    # ==================== STATS ====================

    def get_stats(self) -> dict:
        """Get aggregate statistics for the dashboard."""
        with self.session() as session:
            total_jobs = session.query(Job).count()
            completed_jobs = session.query(Job).filter(Job.status == "COMPLETE").count()
            failed_jobs = session.query(Job).filter(Job.status == "FAILED").count()
            pending_jobs = session.query(Job).filter(
                Job.status.in_(("PENDING", "PARSING", "EXTRACTING", "MATCHING"))
            ).count()

            total_results = session.query(Result).count()
            pass_count = session.query(Result).filter(Result.status == "PASS").count()
            fail_count = session.query(Result).filter(Result.status == "FAIL").count()
            review_count = session.query(Result).filter(Result.status == "REVIEW").count()

            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "pending_jobs": pending_jobs,
                "total_results": total_results,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "review_count": review_count,
                "pass_rate": pass_count / total_results if total_results > 0 else 0.0,
            }
