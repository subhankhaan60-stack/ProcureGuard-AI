"""
API request/response schemas for DocuMatch Architect.

These are API-specific Pydantic models, separate from internal core models.
This allows the API contract to evolve independently from internal data structures.
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


# ==================== SHARED ====================


class ParseInfo(BaseModel):
    """Parse result summary returned in API responses."""

    page_count: int
    tables_found: int
    parse_method: str


class LineItemResponse(BaseModel):
    """Line item as returned by the API."""

    description: str
    quantity: float
    unit_price: float
    total: float


class ValidationIssueResponse(BaseModel):
    """A single validation issue."""

    rule: str
    severity: str
    message: str


# ==================== CONTRACT ====================


class ContractIngestResponse(BaseModel):
    """Response after successfully ingesting a contract."""

    contract_id: str
    vendor_name: str
    contract_type: str
    parse_info: ParseInfo


# ==================== PURCHASE ORDER ====================


class POIngestResponse(BaseModel):
    """Response after successfully ingesting a PO."""

    po_number: str
    vendor_name: str
    order_date: str
    total_amount: float
    currency: str
    line_items: list[LineItemResponse]
    parse_info: ParseInfo


# ==================== INVOICE ====================


class MatchDetailResponse(BaseModel):
    """Result of one match comparison (e.g., Invoice vs PO)."""

    match_type: str
    passed: bool
    score: float
    issues: list[ValidationIssueResponse]


class ValidationSummary(BaseModel):
    """Summary of validation results."""

    status: str = Field(description="PASS, FAIL, or REVIEW")
    matches_passed: int
    total_matches: int
    overall_score: float
    invoice_po_match: Optional[MatchDetailResponse] = None
    invoice_contract_match: Optional[MatchDetailResponse] = None
    po_contract_match: Optional[MatchDetailResponse] = None
    issues: list[ValidationIssueResponse]


class InvoiceProcessResponse(BaseModel):
    """Response after processing and validating an invoice."""

    invoice_number: str
    vendor_name: str
    invoice_date: str
    total_amount: float
    currency: str
    line_items: list[LineItemResponse]
    po_number: Optional[str] = None
    payment_terms: Optional[str] = None
    validation: ValidationSummary


# ==================== HEALTH ====================


class ComponentHealth(BaseModel):
    """Health status of a single system component."""

    status: str = Field(description="ok or error")
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(description="healthy, degraded, or unhealthy")
    components: dict[str, ComponentHealth]


# ==================== PAYMENTS ====================


class PaymentCreateRequest(BaseModel):
    """Record payment data for review; this request never executes a payment."""

    vendor_name: str = Field(min_length=1, max_length=255)
    amount: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    invoice_number: Optional[str] = Field(None, max_length=100)
    invoice_result_id: Optional[int] = Field(None, ge=1)
    payment_reference: Optional[str] = Field(None, max_length=150)
    payment_date: Optional[date] = None
    bank_account_fingerprint: Optional[str] = Field(None, max_length=128)
    source: str = Field(default="MANUAL", pattern=r"^(?i:MANUAL|IMPORT)$")


class PaymentResponse(BaseModel):
    """Recorded payment data, with no implication of payment authorization."""

    id: int
    vendor_name: str
    invoice_number: Optional[str] = None
    invoice_result_id: Optional[int] = None
    payment_reference: Optional[str] = None
    amount: Decimal
    currency: str
    payment_date: Optional[str] = None
    source: str
    status: str
    created_at: Optional[str] = None


# ==================== RESULTS ====================


class ResultResponse(BaseModel):
    """A stored validation result."""

    id: int
    job_id: int
    invoice_file: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    matches_passed: Optional[int] = None
    total_matches: Optional[int] = None
    created_at: str


class StatsResponse(BaseModel):
    """Aggregate statistics for the dashboard."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    total_results: int
    pass_count: int
    fail_count: int
    review_count: int
    pass_rate: float


# ==================== BATCH ====================


class BatchFileRequest(BaseModel):
    """A single file in a batch request."""

    file_path: str = Field(description="Path to invoice PDF file")
    po_number: Optional[str] = Field(None, description="PO number for three-way matching")


class BatchSubmitRequest(BaseModel):
    """Request to submit a batch of invoices for processing."""

    files: list[BatchFileRequest] = Field(
        ..., min_length=1, max_length=100,
        description="List of invoice files to process (max 100)",
    )


class BatchSubmitResponse(BaseModel):
    """Response after submitting a batch."""

    batch_id: int
    total_files: int
    status: str


class BatchErrorDetail(BaseModel):
    """Error detail for a failed file in a batch."""

    file: Optional[str] = None
    error: Optional[str] = None


class BatchStatusResponse(BaseModel):
    """Status of a batch processing job."""

    job_id: int
    status: str
    total_files: int
    completed: int
    failed: int
    pending: int
    processing: int
    eta_seconds: Optional[float] = None
    errors: list[BatchErrorDetail]


# ==================== ERRORS ====================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
