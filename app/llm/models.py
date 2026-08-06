# -*- coding: utf-8 -*-
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class GateResult(BaseModel):
    """Strict result of the direct-answer gate check.

    Contains only `direct_answer` and `reason`. Errors are NOT part of the
    result — any API/auth/quota/timeout/validation failure is raised as
    `LLMError` by the provider.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    direct_answer: StrictBool
    reason: str = Field(min_length=1, max_length=200)
