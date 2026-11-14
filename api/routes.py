
"""
Secure API route handlers for DocuMatch Architect.

Endpoints:
    POST /api/contracts/ingest
    POST /api/pos/ingest
    POST /api/invoices/process
    POST /api/batch/process
    GET  /api/batch/{batch_id}/status
    POST /api/batch/{batch_id}/cancel
    GET  /api/results
    GET  /api/stats
    POST /api/demo/load
    GET  /api/export/excel
    GET  /api/health
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from config import settings
from core.database import Database
from core.exceptions import StoreError
from core.extraction import ExtractionError
from core.demo import load_demo_data
from core.export import export_results_excel
from core.services import (
    BatchService,
    DocumentProcessingError,
    DocumentService,
    MatchService,
)
from core.services.batch_service import BatchFile

from .dependencies import (
    get_batch_service,
    get_database,
    get_document_service,
    get_match_service,
    get_matcher,
    get_vector_store,
    get_po_store,
    require_api_key,
    get_current_user,
    require_roles,
    require_company_user,
    require_company_roles,
)
from .schemas import (
    BatchErrorDetail,
    BatchStatusResponse,
    BatchSubmitRequest,
    BatchSubmitResponse,
    ComponentHealth,
    ContractIngestResponse,
    HealthResponse,
    InvoiceProcessResponse,
    LineItemResponse,
    MatchDetailResponse,
    ParseInfo,
    POIngestResponse,
    LoginRequest,
    CompanyCreateRequest,
    UserCreateRequest,
    PaymentCreateRequest,
    PaymentReviewRequest,
    PaymentResponse,
    ResultResponse,
    StatsResponse,
    ValidationIssueResponse,
    ValidationSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

UPLOAD_CHUNK_SIZE = 1024 * 1024


def _display_filename(filename: str | None) -> str:
    """Return a safe display-only filename."""
    if not filename:
        return "uploaded_document.pdf"

    # Handle both Unix and Windows path separators.
    name = filename.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip(" .")

    return name[:150] or "uploaded_document.pdf"


def _save_uploaded_pdf(
    file: UploadFile,
    destination_dir: Path,
) -> tuple[Path, str]:
    """
    Save a PDF using a server-generated filename.

    Enforces file size and checks the actual PDF header.
    The original user filename is returned only for display metadata.
    """
    display_name = _display_filename(file.filename)

    if not display_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    max_size = settings.max_file_size_bytes

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Never use the user-provided filename as a storage path.
    save_path = destination_dir / f"{uuid4().hex}.pdf"

    total_size = 0
    header = bytearray()

    try:
        with save_path.open("xb") as output:
            while True:
                chunk = file.file.read(UPLOAD_CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the maximum allowed size "
                            f"of {settings.max_file_size_mb} MB."
                        ),
                    )

                if len(header) < 1024:
                    remaining = 1024 - len(header)
                    header.extend(chunk[:remaining])

                output.write(chunk)

        # PDF headers normally appear within the first 1024 bytes.
        if b"%PDF-" not in header:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not contain a valid PDF header.",
            )

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        return save_path, display_name

    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise

    except OSError:
        save_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded PDF")
        raise HTTPException(
            status_code=500,
            detail="Unable to save the uploaded document.",
        )


def _validate_batch_file_path(file_path: str) -> Path:
    """
    Restrict batch processing to existing PDFs inside the invoices directory.
    """
    invoice_root = settings.invoices_path.resolve()

    try:
        requested_path = Path(file_path).resolve(strict=True)
        requested_path.relative_to(invoice_root)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Batch files must be existing files inside the invoices directory.",
        )

    if not requested_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Batch file must be a regular file.",
        )

    if requested_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Batch processing accepts PDF files only.",
        )

    try:
        if requested_path.stat().st_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail="A batch file exceeds the maximum allowed file size.",
            )

        with requested_path.open("rb") as source:
            header = source.read(1024)

        if b"%PDF-" not in header:
            raise HTTPException(
                status_code=400,
                detail="A batch file does not contain a valid PDF header.",
            )

    except HTTPException:
        raise
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="Unable to validate a batch file.",
        )

    return requested_path


# ==================== CONTRACTS ====================


@router.post(
    "/contracts/ingest",
    response_model=ContractIngestResponse,
    summary="Ingest a contract PDF",
    description="Upload and index a contract PDF.",
)
def ingest_contract(
    file: UploadFile = File(..., description="Contract PDF file"),
    vendor_name: str = Form(..., description="Vendor name"),
    contract_type: str = Form("MSA", description="Contract type"),
    service: DocumentService = Depends(get_document_service),
):
    save_path, display_name = _save_uploaded_pdf(
        file,
        settings.contracts_path,
    )

    try:
        contract_id, result = service.ingest_contract(
            file_path=str(save_path),
            vendor_name=vendor_name,
            contract_type=contract_type,
            metadata={"filename": display_name},
        )

        return ContractIngestResponse(
            contract_id=contract_id,
            vendor_name=vendor_name,
            contract_type=contract_type,
            parse_info=ParseInfo(
                page_count=result.page_count,
                tables_found=result.tables_found,
                parse_method=result.parse_method,
            ),
        )

    except (DocumentProcessingError, StoreError):
        save_path.unlink(missing_ok=True)
        raise
    except Exception:
        save_path.unlink(missing_ok=True)
        logger.exception("Contract ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Contract processing failed. Check server logs.",
        )


# ==================== PURCHASE ORDERS ====================


@router.post(
    "/pos/ingest",
    response_model=POIngestResponse,
    summary="Ingest a Purchase Order PDF",
    description="Upload, extract, and index a purchase order PDF.",
)
def ingest_po(
    file: UploadFile = File(..., description="Purchase Order PDF file"),
    vendor_name: str = Form(..., description="Vendor name"),
    model: str | None = Form(None, description="Model override is disabled"),
    service: DocumentService = Depends(get_document_service),
):
    # Do not mutate a shared extraction engine for individual requests.
    if model and model != settings.default_model:
        raise HTTPException(
            status_code=400,
            detail="Model overrides are disabled. Use the configured default model.",
        )

    save_path, display_name = _save_uploaded_pdf(
        file,
        settings.purchase_orders_path,
    )

    try:
        po, result = service.ingest_po(
            file_path=str(save_path),
            vendor_name=vendor_name,
        )

        return POIngestResponse(
            po_number=po.po_number,
            vendor_name=po.vendor_name,
            order_date=po.order_date,
            total_amount=po.total_amount,
            currency=po.currency,
            line_items=[
                LineItemResponse(
                    description=item.description,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    total=item.total,
                )
                for item in po.line_items
            ],
            parse_info=ParseInfo(
                page_count=result.page_count,
                tables_found=result.tables_found,
                parse_method=result.parse_method,
            ),
        )

    except (DocumentProcessingError, ExtractionError, StoreError):
        save_path.unlink(missing_ok=True)
        raise
    except Exception:
        save_path.unlink(missing_ok=True)
        logger.exception("Purchase order ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Purchase order processing failed. Check server logs.",
        )


# ==================== INVOICES ====================


@router.post(
    "/invoices/process",
    response_model=InvoiceProcessResponse,
    summary="Process and validate an invoice",
    description="Upload an invoice PDF and validate it against indexed records.",
)
def process_invoice(
    file: UploadFile = File(..., description="Invoice PDF file"),
    po_number: str | None = Form(None, description="PO number for three-way matching"),
    model: str | None = Form(None, description="Model override is disabled"),
    doc_service: DocumentService = Depends(get_document_service),
    match_service: MatchService = Depends(get_match_service),
):
    if model and model != settings.default_model:
        raise HTTPException(
            status_code=400,
            detail="Model overrides are disabled. Use the configured default model.",
        )

    save_path, display_name = _save_uploaded_pdf(
        file,
        settings.invoices_path,
    )

    try:
        invoice, parse_result = doc_service.process_invoice(str(save_path))

        result = match_service.validate_three_way(
            invoice,
            po_number=po_number,
        )

        def _match_detail(match):
            if match is None:
                return None

            return MatchDetailResponse(
                match_type=match.match_type,
                passed=match.passed,
                score=match.score,
                issues=[
                    ValidationIssueResponse(
                        rule=issue.rule,
                        severity=issue.severity,
                        message=issue.message,
                    )
                    for issue in match.issues
                ],
            )

        return InvoiceProcessResponse(
            invoice_number=invoice.invoice_number,
            vendor_name=invoice.vendor_name,
            invoice_date=invoice.invoice_date,
            total_amount=invoice.total_amount,
            currency=invoice.currency,
            line_items=[
                LineItemResponse(
                    description=item.description,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    total=item.total,
                )
                for item in invoice.line_items
            ],
            po_number=po_number or invoice.po_number,
            payment_terms=invoice.payment_terms,
            validation=ValidationSummary(
                status=result.status,
                matches_passed=result.matches_passed,
                total_matches=result.total_matches,
                overall_score=result.overall_score,
                invoice_po_match=_match_detail(result.invoice_po_match),
                invoice_contract_match=_match_detail(
                    result.invoice_contract_match
                ),
                po_contract_match=_match_detail(result.po_contract_match),
                issues=[
                    ValidationIssueResponse(
                        rule=issue.rule,
                        severity=issue.severity,
                        message=issue.message,
                    )
                    for issue in result.all_issues
                ],
            ),
        )

    except (DocumentProcessingError, ExtractionError):
        save_path.unlink(missing_ok=True)
        raise
    except Exception:
        save_path.unlink(missing_ok=True)
        logger.exception("Invoice processing failed")
        raise HTTPException(
            status_code=500,
            detail="Invoice processing failed. Check server logs.",
        )


# ==================== BATCH PROCESSING ====================


@router.post(
    "/batch/process",
    response_model=BatchSubmitResponse,
    summary="Submit a batch of invoices",
    description="Submit up to 100 existing PDF files from the invoices directory.",
)
def submit_batch(
    request: BatchSubmitRequest,
    batch_service: BatchService = Depends(get_batch_service),
):
    if not request.files:
        raise HTTPException(
            status_code=400,
            detail="At least one batch file is required.",
        )

    if len(request.files) > 100:
        raise HTTPException(
            status_code=400,
            detail="A maximum of 100 files is allowed per batch.",
        )

    files = []

    for item in request.files:
        safe_path = _validate_batch_file_path(item.file_path)

        files.append(
            BatchFile(
                file_path=str(safe_path),
                po_number=item.po_number,
            )
        )

    batch_id = batch_service.submit_batch(files)

    return BatchSubmitResponse(
        batch_id=batch_id,
        total_files=len(files),
        status="PARSING",
    )


@router.get(
    "/batch/{batch_id}/status",
    response_model=BatchStatusResponse,
    summary="Check batch processing status",
)
def get_batch_status(
    batch_id: int,
    batch_service: BatchService = Depends(get_batch_service),
):
    if batch_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid batch ID.")

    status = batch_service.get_batch_status(batch_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Batch not found.")

    return BatchStatusResponse(
        job_id=status.job_id,
        status=status.status,
        total_files=status.total_files,
        completed=status.completed,
        failed=status.failed,
        pending=status.pending,
        processing=status.processing,
        eta_seconds=status.eta_seconds,
        errors=[
            BatchErrorDetail(file=e["file"], error=e["error"])
            for e in status.errors
        ],
    )


@router.post(
    "/batch/{batch_id}/cancel",
    summary="Cancel a running batch",
)
def cancel_batch(
    batch_id: int,
    batch_service: BatchService = Depends(get_batch_service),
):
    if batch_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid batch ID.")

    if batch_service.cancel_batch(batch_id):
        return {"message": "Batch cancellation requested."}

    raise HTTPException(
        status_code=404,
        detail="Batch not found or already finished.",
    )


# ==================== RESULTS ====================


@router.get(
    "/results",
    response_model=list[ResultResponse],
    summary="Query validation results",
)
def get_results(
    vendor_name: str | None = Query(None),
    status: str | None = Query(None),
    job_id: int | None = Query(None, ge=1),
    limit: int = Query(50, ge=1, le=500),
    db: Database = Depends(get_database),
):
    results = db.get_results(
        vendor_name=vendor_name,
        status=status,
        job_id=job_id,
        limit=limit,
    )

    return [
        ResultResponse(
            id=r.id,
            job_id=r.job_id,
            invoice_file=r.invoice_file,
            vendor_name=r.vendor_name,
            invoice_number=r.invoice_number,
            status=r.status,
            confidence=r.confidence,
            matches_passed=r.matches_passed,
            total_matches=r.total_matches,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in results
    ]



# ==================== USER AUTHENTICATION ====================

@router.post("/auth/login", summary="Exchange user credentials for a short-lived access token")
def login_user(
    payload: LoginRequest,
    request: Request,
    db: Database = Depends(get_database),
):
    """Authenticate with persistent, per-email-and-client-IP throttling."""
    from core.auth import create_access_token

    client_ip = request.client.host if request.client else "unknown"
    retry_after = db.login_rate_limit_remaining(payload.email, client_ip)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.authenticate_user(payload.email, payload.password)
    if user is None:
        db.record_failed_login(payload.email, client_ip)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    db.clear_login_attempts(payload.email, client_ip)
    if not settings.user_session_secret or len(settings.user_session_secret) < 32:
        raise HTTPException(status_code=503, detail="User sessions are not configured. Set USER_SESSION_SECRET to a random secret of at least 32 characters.")
    token = create_access_token(user, settings.user_session_secret, settings.user_session_ttl_seconds)
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.user_session_ttl_seconds, "user": user}


@router.post("/companies", summary="Create a company tenant (administrator only)")
def create_company_account(
    payload: CompanyCreateRequest,
    db: Database = Depends(get_database),
    admin: dict = Depends(require_roles("ADMIN")),
):
    """Create a tenant container; existing unassigned records are not migrated automatically."""
    try:
        return db.create_company(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/auth/users", summary="Create a user account (administrator only)")
def create_user_account(
    payload: UserCreateRequest,
    db: Database = Depends(get_database),
    admin: dict = Depends(require_roles("ADMIN")),
):
    """Provision an account using an already authenticated administrator."""
    try:
        return db.create_user(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/auth/me", summary="Return the authenticated user profile")
def get_authenticated_profile(user: dict = Depends(get_current_user)):
    return user


# ==================== STATISTICS ====================


@router.post(
    "/payments",
    response_model=PaymentResponse,
    summary="Record payment data for human review",
    description="Stores a payment record in PENDING_REVIEW. This endpoint does not approve, execute, or authorize payment.",
)
def create_payment(
    payload: PaymentCreateRequest,
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_roles("ADMIN", "PREPARER")),
):
    try:
        return db.create_payment_record(
            **payload.model_dump(),
            created_by_user_id=user["id"],
            company_id=user["company_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/payments",
    response_model=list[PaymentResponse],
    summary="List recorded payment data",
    description="Returns recorded payment entries. PENDING_REVIEW is the only supported status in this initial release.",
)
def list_payments(
    vendor_name: str | None = Query(None, min_length=1, max_length=255),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_user),
):
    try:
        return db.list_payment_records(
            vendor_name=vendor_name,
            status=status,
            limit=limit,
            company_id=user["company_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/payments/{payment_id}/review",
    summary="Record a human review disposition for a payment record",
    description=(
        "Records CLEAR or REJECT with a note. This does not authorize, execute, "
        "block, or reverse a bank transfer. Reviewer identity is taken from the authenticated account."
    ),
)
def review_payment(
    payment_id: int,
    payload: PaymentReviewRequest,
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_roles("ADMIN", "REVIEWER", "APPROVER")),
):
    try:
        return db.review_payment_record(
            payment_id=payment_id,
            decision=payload.decision,
            reviewer_reference=user["email"],
            note=payload.note,
            reviewer_user_id=user["id"],
            company_id=user["company_id"],
        )
    except ValueError as exc:
        message = str(exc)
        status_code = (
            404 if "not found" in message.lower()
            else 409 if "Only PENDING_REVIEW" in message or "cannot review their own payment" in message
            else 422
        )
        raise HTTPException(status_code=status_code, detail=message) from exc


@router.get(
    "/payments/{payment_id}/reviews",
    summary="List review history for a payment record",
)
def list_payment_reviews(
    payment_id: int,
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_user),
):
    try:
        return db.list_payment_review_events(payment_id, company_id=user["company_id"])
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get(
    "/risk/payment-review-queue",
    summary="Build a payment review queue from pending records and risk indicators",
    description=(
        "Combines pending payment records with duplicate and vendor bank-profile "
        "review indicators. It is read-only and does not approve, block, or execute payments."
    ),
)
def get_payment_review_queue(
    vendor_name: str | None = Query(None, min_length=1, max_length=255),
    limit: int = Query(100, ge=1, le=500),
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_roles("ADMIN", "REVIEWER", "APPROVER", "AUDITOR")),
):
    try:
        pending = db.list_payment_records(
            vendor_name=vendor_name,
            status="PENDING_REVIEW",
            limit=limit,
            company_id=user["company_id"],
        )
        duplicate_findings = db.find_duplicate_payment_candidates(
            vendor_name=vendor_name,
            limit=limit,
            company_id=user["company_id"],
        )
        bank_findings = db.find_payment_bank_profile_mismatches(
            vendor_name=vendor_name,
            limit=limit,
            company_id=user["company_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    flags_by_payment: dict[int, list[str]] = {}
    for finding in duplicate_findings:
        for payment_id in finding.get("payment_ids", []):
            flags_by_payment.setdefault(payment_id, []).append(
                finding.get("rule", "duplicate_payment_candidate")
            )
    for finding in bank_findings:
        payment_id = finding.get("payment_id")
        if payment_id is not None:
            flags_by_payment.setdefault(payment_id, []).append(
                finding.get("rule", "vendor_bank_profile_mismatch")
            )

    queue = []
    for payment in pending:
        payment_id = payment.get("id")
        flags = sorted(set(flags_by_payment.get(payment_id, [])))
        queue.append({
            **payment,
            "risk_flags": flags,
            "risk_review_required": bool(flags),
        })

    return {
        "queue_type": "pending_payment_human_review",
        "record_count": len(queue),
        "flagged_count": sum(1 for item in queue if item["risk_review_required"]),
        "records": queue,
        "risk_findings": {
            "duplicate_payment_candidates": duplicate_findings,
            "vendor_bank_profile_mismatches": bank_findings,
        },
        "disclaimer": (
            "Risk flags are indicators for human verification, not proof of fraud. "
            "This read-only queue does not approve, block, execute, or reverse payments. "
            "A payment record marked REVIEWED_CLEAR is not a bank instruction or proof "
            "of organizational approval."
        ),
    }


@router.get(
    "/risk/payment-bank-mismatches",
    summary="Find payment records with changed vendor bank fingerprints",
    description="Compares supplied payment fingerprints with registered vendor profiles. Results require independent human verification and do not block or authorize payments.",
)
def get_payment_bank_mismatches(
    vendor_name: str | None = Query(None, min_length=1, max_length=255),
    limit: int = Query(100, ge=1, le=500),
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_roles("ADMIN", "REVIEWER", "APPROVER", "AUDITOR")),
):
    try:
        findings = db.find_payment_bank_profile_mismatches(
            vendor_name=vendor_name,
            limit=limit,
            company_id=user["company_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "analysis_type": "payment_bank_profile_mismatch_review",
        "finding_count": len(findings),
        "findings": findings,
        "disclaimer": (
            "Fingerprint mismatches are review indicators, not proof of fraud. "
            "Verify beneficiary details through an independently trusted channel. "
            "This endpoint does not block, authorize, execute, or reverse payments. "
            "Records without a fingerprint or registered baseline cannot be compared."
        ),
    }


@router.get(
    "/risk/payment-duplicates",
    summary="Find possible duplicate payment records",
    description="Returns candidate duplicate indicators for human review; it does not block, reverse, or authorize payments.",
)
def get_payment_duplicate_candidates(
    vendor_name: str | None = Query(None, min_length=1, max_length=255),
    limit: int = Query(100, ge=1, le=500),
    db: Database = Depends(get_database),
    user: dict = Depends(require_company_roles("ADMIN", "REVIEWER", "APPROVER", "AUDITOR")),
):
    try:
        findings = db.find_duplicate_payment_candidates(
            vendor_name=vendor_name,
            limit=limit,
            company_id=user["company_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "analysis_type": "payment_duplicate_review",
        "finding_count": len(findings),
        "findings": findings,
        "disclaimer": (
            "Candidate indicators only. Repeated invoice payments may be legitimate "
            "installments, and payment references may be reused. Human verification "
            "is required; payments are not blocked or authorized, and none are "
            "reversed by this endpoint."
        ),
    }


@router.get(
    "/results/{result_id}/evidence",
    summary="Retrieve validation explanation and supporting evidence",
)
def get_result_evidence(
    result_id: int,
    db: Database = Depends(get_database),
):
    evidence = db.get_result_evidence(result_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return evidence


@router.get(
    "/risk/invoice-patterns",
    summary="Find repeated same-value invoice patterns",
    description="Returns review indicators from stored invoice results; it does not detect or authorize payments.",
)
def get_invoice_patterns(
    vendor_name: str | None = Query(None, min_length=1, max_length=255),
    lookback_days: int = Query(7, ge=1, le=365),
    min_occurrences: int = Query(3, ge=2, le=20),
    db: Database = Depends(get_database),
):
    findings = db.find_suspicious_invoice_patterns(
        lookback_days=lookback_days,
        min_occurrences=min_occurrences,
        vendor_name=vendor_name,
    )
    return {
        "analysis_type": "invoice_pattern_review",
        "lookback_basis": "result processing timestamp, not invoice date",
        "lookback_days": lookback_days,
        "min_occurrences": min_occurrences,
        "finding_count": len(findings),
        "findings": findings,
        "disclaimer": "Indicators only. Repeated amounts may be legitimate recurring invoices. Human review is required; no payment records or payment authorization are involved.",
    }


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Processing statistics",
)
def get_stats(db: Database = Depends(get_database)):
    return StatsResponse(**db.get_stats())


# ==================== DEMO ====================


@router.post(
    "/demo/load",
    summary="Load sample demo data",
)
def load_demo(db: Database = Depends(get_database)):
    summary = load_demo_data(
        vector_store=get_vector_store(),
        po_store=get_po_store(),
        matcher=get_matcher(),
        database=db,
    )

    return {
        "message": "Demo data loaded successfully",
        "contracts_indexed": summary["contracts_indexed"],
        "pos_indexed": summary["pos_indexed"],
        "invoices_processed": summary["invoices_processed"],
        "results": summary["results"],
    }


# ==================== EXPORT ====================


@router.get(
    "/export/excel",
    summary="Export results as Excel",
)
def export_excel(
    vendor_name: str | None = Query(None),
    status: str | None = Query(None),
    db: Database = Depends(get_database),
):
    excel_bytes = export_results_excel(
        database=db,
        vendor_name=vendor_name,
        status=status,
    )

    filename = (
        f"documatch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    return Response(
        content=excel_bytes,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ==================== HEALTH ====================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
def health_check(
    doc_service: DocumentService = Depends(get_document_service),
):
    components = {}

    parser = doc_service.parser
    parser_name = "Docling" if parser._docling_available else "pdfplumber"

    components["parser"] = ComponentHealth(
        status="ok",
        detail=parser_name,
    )

    engine = doc_service.extraction_engine

    try:
        ollama_ok, _ = engine.check_connection()
        components["ollama"] = ComponentHealth(
            status="ok" if ollama_ok else "error",
            detail="Connected" if ollama_ok else "Unavailable",
        )
    except Exception:
        logger.exception("Ollama health check failed")
        components["ollama"] = ComponentHealth(
            status="error",
            detail="Health check failed",
        )

    try:
        doc_service.vector_store.get_stats()
        components["chromadb"] = ComponentHealth(
            status="ok",
            detail="Connected",
        )
    except Exception:
        logger.exception("ChromaDB health check failed")
        components["chromadb"] = ComponentHealth(
            status="error",
            detail="Health check failed",
        )

    statuses = [component.status for component in components.values()]

    if all(status == "ok" for status in statuses):
        overall = "healthy"
    elif components["parser"].status == "ok":
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        components=components,
    )
