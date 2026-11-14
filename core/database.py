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
import hashlib
from datetime import date, datetime, time, timezone, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Boolean,
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
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
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


class Company(Base):
    """Tenant account that owns application users and business records."""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(80), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class User(Base):
    """Local application identity; passwords are stored only as salted PBKDF2 hashes."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
    email = Column(String(320), nullable=False, unique=True, index=True)
    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(30), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class LoginAttempt(Base):
    """Persistent throttling state keyed by a one-way email/IP digest."""

    __tablename__ = "login_attempts"

    key = Column(String(64), primary_key=True)
    attempts = Column(Integer, nullable=False, default=0)
    window_started_at = Column(DateTime, nullable=False)
    locked_until = Column(DateTime, nullable=True)


class VendorBankProfile(Base):
    """Current vendor bank identity; never stores a full account number."""

    __tablename__ = "vendor_bank_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
    vendor_key = Column(String(255), nullable=False, index=True)
    vendor_name = Column(String(255), nullable=False)
    bank_name = Column(String(255), nullable=False)
    account_last4 = Column(String(4), nullable=False)
    account_fingerprint = Column(String(128), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class Payment(Base):
    """A recorded payment instruction or payment event; never stores raw bank details."""

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
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
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PaymentReviewEvent(Base):
    """Append-only record of a human review disposition for a payment entry."""

    __tablename__ = "payment_review_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(Integer, ForeignKey("payments.id"), nullable=False, index=True)
    decision = Column(String(30), nullable=False)
    reviewer_reference = Column(String(128), nullable=False)
    note = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    """Audit trail for significant system actions."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
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

            # Backward-compatible tenant ownership for business records.
            # Existing rows remain NULL until an administrator verifies ownership.
            for table_name in ("jobs", "vendor_bank_profiles", "payments", "audit_log"):
                columns = {
                    row[1] for row in connection.exec_driver_sql(
                        f"PRAGMA table_info({table_name})"
                    ).fetchall()
                }
                if "company_id" not in columns:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table_name} ADD COLUMN company_id INTEGER REFERENCES companies(id)"
                    )
                    connection.exec_driver_sql(
                        f"CREATE INDEX IF NOT EXISTS ix_{table_name}_company_id ON {table_name}(company_id)"
                    )

            # Backward-compatible tenant ownership for user identities.
            user_columns = {
                row[1] for row in connection.exec_driver_sql("PRAGMA table_info(users)").fetchall()
            }
            if "company_id" not in user_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN company_id INTEGER REFERENCES companies(id)"
                )
                connection.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_users_company_id ON users(company_id)"
                )

            # Backward-compatible migration for payment creator attribution.
            payment_columns = {
                row[1] for row in connection.exec_driver_sql("PRAGMA table_info(payments)").fetchall()
            }
            if "created_by_user_id" not in payment_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE payments ADD COLUMN created_by_user_id INTEGER REFERENCES users(id)"
                )
                connection.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_payments_created_by_user_id ON payments(created_by_user_id)"
                )

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

    # ==================== USER IDENTITY FOUNDATION ====================

    def create_company(self, name: str, slug: str) -> dict:
        """Create a tenant company with a stable, unique URL-safe slug."""
        import re

        name_value = (name or "").strip()
        slug_value = (slug or "").strip().lower()
        if not name_value or len(name_value) > 255:
            raise ValueError("Company name is required and must be 255 characters or fewer.")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?", slug_value):
            raise ValueError("Company slug must contain 1-80 lowercase letters, numbers, or internal hyphens.")

        with self.session() as session:
            if session.query(Company).filter(Company.slug == slug_value).first():
                raise ValueError("A company with this slug already exists.")
            company = Company(name=name_value, slug=slug_value, is_active=True)
            session.add(company)
            session.flush()
            self._audit(session, "company_created", "company", str(company.id), {
                "name": name_value, "slug": slug_value,
            })
            session.commit()
            session.refresh(company)
            return self._company_dict(company)

    @staticmethod
    def _company_dict(company: Company) -> dict:
        return {
            "id": company.id,
            "name": company.name,
            "slug": company.slug,
            "is_active": company.is_active,
            "created_at": company.created_at.isoformat() if company.created_at else None,
        }

    def get_company(self, company_id: int) -> Optional[dict]:
        if company_id <= 0:
            return None
        with self.session() as session:
            company = session.get(Company, company_id)
            return self._company_dict(company) if company else None

    def create_user(
        self,
        email: str,
        full_name: str,
        password: str,
        role: str = "REVIEWER",
        company_id: Optional[int] = None,
    ) -> dict:
        """Create a local user account. Provisioning authorization is not yet exposed by the API."""
        from core.auth import hash_password

        email_value = (email or "").strip().casefold()
        name_value = (full_name or "").strip()
        role_value = (role or "").strip().upper()
        if not email_value or len(email_value) > 320 or "@" not in email_value:
            raise ValueError("A valid email address is required.")
        if not name_value or len(name_value) > 255:
            raise ValueError("Full name is required and must be 255 characters or fewer.")
        if role_value not in ("ADMIN", "PREPARER", "REVIEWER", "APPROVER", "AUDITOR"):
            raise ValueError("Unsupported user role.")
        if not isinstance(password, str) or len(password) < 12 or len(password) > 256:
            raise ValueError("Password must be between 12 and 256 characters.")

        with self.session() as session:
            if company_id is not None:
                company = session.get(Company, company_id)
                if company is None or not company.is_active:
                    raise ValueError("A valid active company is required.")
            if session.query(User).filter(User.email == email_value).first():
                raise ValueError("A user with this email already exists.")
            user = User(
                company_id=company_id,
                email=email_value,
                full_name=name_value,
                password_hash=hash_password(password),
                role=role_value,
                is_active=True,
            )
            session.add(user)
            session.flush()
            self._audit(session, "user_created", "user", str(user.id), {
                "email": email_value, "role": role_value,
                "company_id": company_id,
            })
            session.commit()
            session.refresh(user)
            return self._user_dict(user)

    @staticmethod
    def _user_dict(user: User) -> dict:
        return {
            "id": user.id,
            "company_id": user.company_id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    @staticmethod
    def _login_attempt_key(email: str, client_ip: str) -> str:
        """Hash normalized identifiers; never persist raw email addresses or IPs."""
        normalized_email = (email or "").strip().casefold()
        normalized_ip = (client_ip or "unknown").strip()
        return hashlib.sha256(
            f"{normalized_email}\\0{normalized_ip}".encode("utf-8")
        ).hexdigest()

    def login_rate_limit_remaining(
        self,
        email: str,
        client_ip: str,
        now: Optional[datetime] = None,
    ) -> int:
        """Return remaining lockout seconds; defaults to 5 failures / 15 minutes."""
        current = now or datetime.now(timezone.utc)
        key = self._login_attempt_key(email, client_ip)
        with self.session() as session:
            state = session.get(LoginAttempt, key)
            if not state or not state.locked_until:
                return 0
            locked_until = state.locked_until
            if locked_until.tzinfo is None:
                locked_until = locked_until.replace(tzinfo=timezone.utc)
            return max(0, int((locked_until - current).total_seconds()))

    def record_failed_login(
        self,
        email: str,
        client_ip: str,
        now: Optional[datetime] = None,
    ) -> int:
        """Atomically record a failure and return active lockout seconds."""
        current = now or datetime.now(timezone.utc)
        key = self._login_attempt_key(email, client_ip)
        window_seconds = 15 * 60
        threshold = 5
        lockout_seconds = 15 * 60
        current_naive = current.astimezone(timezone.utc).replace(tzinfo=None)

        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                row = connection.exec_driver_sql(
                    "SELECT attempts, window_started_at, locked_until "
                    "FROM login_attempts WHERE key = ?", (key,)
                ).fetchone()
                if row and row[2]:
                    locked_until = datetime.fromisoformat(row[2])
                    if locked_until.tzinfo is None:
                        locked_until = locked_until.replace(tzinfo=timezone.utc)
                    remaining = int((locked_until - current).total_seconds())
                    if remaining > 0:
                        connection.commit()
                        return remaining

                attempts = 0
                reset_window = True
                if row:
                    started = datetime.fromisoformat(row[1])
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    if (current - started).total_seconds() < window_seconds:
                        attempts = int(row[0])
                        reset_window = False

                attempts += 1
                window_started = current_naive if reset_window else datetime.fromisoformat(row[1])
                locked_until_value = (
                    (current + timedelta(seconds=lockout_seconds))
                    .astimezone(timezone.utc).replace(tzinfo=None).isoformat()
                    if attempts >= threshold else None
                )
                connection.exec_driver_sql(
                    "INSERT INTO login_attempts "
                    "(key, attempts, window_started_at, locked_until) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET attempts=excluded.attempts, "
                    "window_started_at=excluded.window_started_at, locked_until=excluded.locked_until",
                    (key, attempts, window_started.isoformat(), locked_until_value),
                )
                connection.commit()
                return lockout_seconds if locked_until_value else 0
            except Exception:
                connection.rollback()
                raise

    def clear_login_attempts(self, email: str, client_ip: str) -> None:
        """Clear the matching throttling record after successful authentication."""
        key = self._login_attempt_key(email, client_ip)
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "DELETE FROM login_attempts WHERE key = ?", (key,)
            )

    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Verify local credentials and return non-secret identity data."""
        from core.auth import verify_password

        email_value = (email or "").strip().casefold()
        if not email_value or not isinstance(password, str):
            return None
        with self.session() as session:
            user = session.query(User).filter(User.email == email_value).first()
            if user is None or not user.is_active:
                return None
            if not verify_password(password, user.password_hash):
                return None
            return self._user_dict(user)

    def get_user(self, user_id: int) -> Optional[dict]:
        if user_id <= 0:
            return None
        with self.session() as session:
            user = session.get(User, user_id)
            return self._user_dict(user) if user else None

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
        created_by_user_id: Optional[int] = None,
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
                created_by_user_id=created_by_user_id,
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
                "created_by_user_id": created_by_user_id,
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
            "created_by_user_id": payment.created_by_user_id,
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

    def review_payment_record(
        self,
        payment_id: int,
        decision: str,
        reviewer_reference: str,
        note: str,
        reviewer_user_id: Optional[int] = None,
    ) -> dict:
        """
        Record a review disposition. This is not a bank instruction or transfer.

        When authenticated user IDs are available, enforce creator/reviewer
        separation independently of the display-only email reference.
        """
        decision_value = (decision or "").strip().upper()
        reviewer_value = (reviewer_reference or "").strip()
        note_value = (note or "").strip()
        if payment_id <= 0:
            raise ValueError("payment_id must be a positive integer.")
        if decision_value not in ("CLEAR", "REJECT"):
            raise ValueError("decision must be CLEAR or REJECT.")
        if not reviewer_value or len(reviewer_value) > 128:
            raise ValueError("reviewer_reference is required and must be 128 characters or fewer.")
        if not note_value or len(note_value) > 1000:
            raise ValueError("A review note is required and must be 1000 characters or fewer.")

        final_status = "REVIEWED_CLEAR" if decision_value == "CLEAR" else "REVIEWED_REJECTED"
        with self.session() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            payment = session.get(Payment, payment_id)
            if payment is None:
                raise ValueError("Payment record was not found.")
            if payment.status != "PENDING_REVIEW":
                raise ValueError("Only PENDING_REVIEW payment records can receive a review decision.")
            if (
                payment.created_by_user_id is not None
                and reviewer_user_id is not None
                and payment.created_by_user_id == reviewer_user_id
            ):
                raise ValueError("Payment creator cannot review their own payment (segregation of duties).")

            payment.status = final_status
            event = PaymentReviewEvent(
                payment_id=payment.id,
                decision=decision_value,
                reviewer_reference=reviewer_value,
                note=note_value,
            )
            session.add(event)
            session.flush()
            self._audit(session, "payment_review_recorded", "payment", str(payment.id), {
                "review_event_id": event.id,
                "decision": decision_value,
                "reviewer_reference": reviewer_value,
                "reviewer_user_id": reviewer_user_id,
                "status": final_status,
                "payment_execution_authorized": False,
            })
            session.commit()
            session.refresh(payment)
            return {
                "payment": self._payment_dict(payment),
                "review": {
                    "id": event.id,
                    "payment_id": event.payment_id,
                    "decision": event.decision,
                    "reviewer_reference": event.reviewer_reference,
                    "note": event.note,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                },
                "payment_execution_authorized": False,
            }

    def list_payment_review_events(self, payment_id: int) -> list[dict]:
        if payment_id <= 0:
            raise ValueError("payment_id must be a positive integer.")
        with self.session() as session:
            if session.get(Payment, payment_id) is None:
                raise ValueError("Payment record was not found.")
            events = session.query(PaymentReviewEvent).filter(
                PaymentReviewEvent.payment_id == payment_id
            ).order_by(PaymentReviewEvent.created_at.asc(), PaymentReviewEvent.id.asc()).all()
            return [{
                "id": event.id,
                "payment_id": event.payment_id,
                "decision": event.decision,
                "reviewer_reference": event.reviewer_reference,
                "note": event.note,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            } for event in events]

    def find_payment_bank_profile_mismatches(
        self,
        vendor_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Flag recorded payment entries whose supplied keyed bank fingerprint
        differs from the currently registered vendor bank profile.

        This is a review indicator only. It neither changes the profile nor
        blocks, authorizes, or executes a payment. Entries without a supplied
        fingerprint or without a registered baseline cannot be compared.
        """
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")

        with self.session() as session:
            query = session.query(Payment, VendorBankProfile).join(
                VendorBankProfile,
                func.lower(func.trim(Payment.vendor_name)) == VendorBankProfile.vendor_key,
            ).filter(
                Payment.bank_account_fingerprint.isnot(None),
                Payment.bank_account_fingerprint != "",
            )
            if vendor_name and vendor_name.strip():
                query = query.filter(
                    func.upper(func.trim(Payment.vendor_name))
                    == vendor_name.strip().upper()
                )
            rows = query.order_by(
                Payment.created_at.desc(), Payment.id.desc()
            ).limit(5000).all()

        findings = []
        for payment, profile in rows:
            if payment.bank_account_fingerprint == profile.account_fingerprint:
                continue
            findings.append({
                "rule": "payment_bank_profile_mismatch",
                "status": "REVIEW",
                "payment_id": payment.id,
                "vendor_name": payment.vendor_name,
                "invoice_number": payment.invoice_number,
                "amount": format(Decimal(str(payment.amount)).quantize(Decimal("0.01")), ".2f"),
                "currency": payment.currency,
                "payment_reference": payment.payment_reference,
                "registered_bank_name": profile.bank_name,
                "registered_account_last4": profile.account_last4,
                "profile_updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
                "explanation": (
                    "The keyed bank-account fingerprint supplied with this payment "
                    "record differs from the current registered vendor bank profile. "
                    "Verify the beneficiary directly using an independently trusted "
                    "contact channel before any payment action."
                ),
                "payment_authorized": False,
            })
            if len(findings) >= limit:
                break
        return findings

    def find_duplicate_payment_candidates(
        self,
        vendor_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Identify possible duplicate payment records for human review.

        Rules:
        - Same normalized vendor + currency + non-empty payment reference.
        - Same normalized vendor + invoice number + amount + currency.

        These are candidate indicators only. Repeated invoice payments may be
        legitimate installments, and payment references may be reused by some
        source systems. No record is changed and no fraud conclusion is made.
        """
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")

        with self.session() as session:
            query = session.query(Payment)
            if vendor_name and vendor_name.strip():
                query = query.filter(
                    func.upper(func.trim(Payment.vendor_name))
                    == vendor_name.strip().upper()
                )
            # Bound the working set so a large local database cannot cause an
            # unbounded in-memory scan. Newest records are examined first.
            payments = query.order_by(
                Payment.created_at.desc(), Payment.id.desc()
            ).limit(5000).all()

        groups = {}
        for payment in payments:
            vendor_key = (payment.vendor_name or "").strip().casefold()
            currency = (payment.currency or "").strip().upper()
            if not vendor_key or not currency:
                continue

            reference = (payment.payment_reference or "").strip().casefold()
            if reference:
                key = ("duplicate_payment_reference", vendor_key, currency, reference)
                groups.setdefault(key, []).append(payment)

            invoice = (payment.invoice_number or "").strip().casefold()
            if invoice:
                amount = Decimal(str(payment.amount)).quantize(Decimal("0.01"))
                key = (
                    "repeated_invoice_payment", vendor_key, currency,
                    invoice, amount,
                )
                groups.setdefault(key, []).append(payment)

        findings = []
        for key, records in groups.items():
            if len(records) < 2:
                continue
            rule = key[0]
            ordered = sorted(records, key=lambda item: (item.created_at, item.id))
            first = ordered[0]
            if rule == "duplicate_payment_reference":
                explanation = (
                    "The same non-empty payment reference appears more than once "
                    "for this vendor and currency. Verify the source statement and "
                    "bank reference; some source systems may reuse references."
                )
            else:
                explanation = (
                    "The same vendor, invoice number, amount, and currency appear "
                    "in multiple payment records. This may represent a duplicate "
                    "entry or legitimate split/installment payments and requires "
                    "human verification."
                )
            findings.append({
                "rule": rule,
                "status": "REVIEW",
                "vendor_name": first.vendor_name,
                "invoice_number": first.invoice_number if rule == "repeated_invoice_payment" else None,
                "amount": format(Decimal(str(first.amount)).quantize(Decimal("0.01")), ".2f")
                    if rule == "repeated_invoice_payment" else None,
                "currency": first.currency,
                "payment_reference": first.payment_reference
                    if rule == "duplicate_payment_reference" else None,
                "occurrences": len(records),
                "payment_ids": [item.id for item in ordered],
                "evidence": [
                    {
                        "payment_id": item.id,
                        "invoice_number": item.invoice_number,
                        "payment_reference": item.payment_reference,
                        "amount": format(Decimal(str(item.amount)).quantize(Decimal("0.01")), ".2f"),
                        "currency": item.currency,
                        "payment_date": item.payment_date.date().isoformat()
                            if item.payment_date else None,
                        "source": item.source,
                        "status": item.status,
                    }
                    for item in ordered
                ],
                "explanation": explanation,
            })

        findings.sort(
            key=lambda item: (-item["occurrences"], item["vendor_name"].casefold(), item["rule"])
        )
        return findings[:limit]

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
