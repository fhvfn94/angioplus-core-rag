# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from typing import Optional


class LLMErrorType(str, Enum):
    """Provider-agnostic classification of LLM failures."""

    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    TIMEOUT_OR_NETWORK = "timeout_or_network"
    TEMPORARY_UNAVAILABLE = "temporary_unavailable"
    INVALID_RESPONSE = "invalid_response"
    UNEXPECTED_INTERNAL = "unexpected_internal"


# Error types that map to HTTP 200 "temporarily unavailable" in the /ask
# contract. AUTHENTICATION and UNEXPECTED_INTERNAL stay as HTTP 500.
TRANSIENT_ERROR_TYPES = {
    LLMErrorType.QUOTA_EXCEEDED,
    LLMErrorType.TIMEOUT_OR_NETWORK,
    LLMErrorType.TEMPORARY_UNAVAILABLE,
    LLMErrorType.INVALID_RESPONSE,
}


def is_transient_error(error_type: LLMErrorType) -> bool:
    """Return True if an LLMError should map to HTTP 200 (temporarily unavailable)."""
    return error_type in TRANSIENT_ERROR_TYPES


class LLMError(Exception):
    """A single, provider-agnostic error type for LLM operations.

    `message` must never contain credentials or secrets. `status_code` is the
    HTTP/API code if available, otherwise None.
    """

    def __init__(
        self,
        error_type: LLMErrorType,
        message: str = "",
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.status_code = status_code

    def __repr__(self) -> str:
        return (
            f"LLMError({self.error_type.value!r}, "
            f"message={self.message!r}, status_code={self.status_code!r})"
        )
