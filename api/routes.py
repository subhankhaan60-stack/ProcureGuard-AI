
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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
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


# ==================== STATISTICS ====================


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
