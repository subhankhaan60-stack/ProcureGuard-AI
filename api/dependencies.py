"""
FastAPI dependency injection for DocuMatch Architect.

Creates and caches service instances shared across requests.
Uses functools.lru_cache for singleton behavior.
"""

from functools import lru_cache
import hmac

from fastapi import Header, HTTPException, status

from config import settings
from core.database import Database
from core.extraction import ExtractionEngine
from core.matcher import Matcher
from core.parser_engine import ParserEngine
from core.po_store import POStore
from core.services import BatchService, DocumentService, MatchService
from core.vector_store import VectorStore


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require the configured shared API key before accessing API routes.

    This is a deployment-level access gate, not user identity or tenant isolation.
    """
    expected = settings.api_auth_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured. Set API_AUTH_TOKEN before use.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@lru_cache
def get_parser() -> ParserEngine:
    return ParserEngine(fallback_enabled=settings.parser_fallback_enabled)


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(
        persist_directory=str(settings.chroma_path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_po_store() -> POStore:
    return POStore(persist_directory=str(settings.chroma_path))


@lru_cache
def get_extraction_engine() -> ExtractionEngine:
    return ExtractionEngine(
        model=settings.default_model,
        ollama_host=settings.ollama_host,
    )


@lru_cache
def get_matcher() -> Matcher:
    return Matcher(
        vector_store=get_vector_store(),
        po_store=get_po_store(),
        ollama_host=settings.ollama_host,
        model=settings.default_model,
    )


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        parser=get_parser(),
        vector_store=get_vector_store(),
        po_store=get_po_store(),
        extraction_engine=get_extraction_engine(),
    )


@lru_cache
def get_match_service() -> MatchService:
    return MatchService(matcher=get_matcher(), po_store=get_po_store())


@lru_cache
def get_database() -> Database:
    return Database(db_path=settings.db_path)


@lru_cache
def get_batch_service() -> BatchService:
    return BatchService(
        document_service=get_document_service(),
        match_service=get_match_service(),
        database=get_database(),
    )
