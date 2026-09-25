"""Pydantic data contracts exchanged between agents, the API and storage.

Every agent consumes and produces these models, which guarantees structured,
validated JSON flows between nodes of the LangGraph workflow.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏ -‮⁠-⁤﻿]")


def clean_text(value: str) -> str:
    """Normalise user-supplied text: NFKC, strip control/zero-width chars, trim.

    Args:
        value: Raw text.

    Returns:
        The cleaned text.
    """
    value = unicodedata.normalize("NFKC", value)
    value = _CONTROL_CHARS.sub("", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


class Severity(str, Enum):
    """Threat severity levels, ordered from least to most severe."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Numeric rank (0 = LOW, 3 = CRITICAL) used for comparisons."""
        return list(Severity).index(self)

    @classmethod
    def from_rank(cls, rank: int) -> "Severity":
        """Return the severity for a numeric rank, clamped to the valid range.

        Args:
            rank: Numeric rank.

        Returns:
            The matching :class:`Severity`.
        """
        members = list(cls)
        return members[max(0, min(rank, len(members) - 1))]


class IncidentCategory(str, Enum):
    """High-level incident categories."""

    CYBER_ATTACK = "CYBER_ATTACK"
    NATURAL_DISASTER = "NATURAL_DISASTER"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    PUBLIC_SAFETY = "PUBLIC_SAFETY"
    OTHER = "OTHER"


# --------------------------------------------------------------------------- #
# API input
# --------------------------------------------------------------------------- #
class IncidentSubmission(BaseModel):
    """Raw incident report submitted to the crisis center."""

    model_config = ConfigDict(extra="forbid")

    report: str = Field(
        ...,
        min_length=20,
        max_length=10_000,
        description="Free-text incident report as received from the field.",
    )
    title: str | None = Field(default=None, max_length=200, description="Optional short title.")
    location: str | None = Field(default=None, max_length=200)
    reported_by: str | None = Field(default=None, max_length=120)
    source: str | None = Field(default=None, max_length=60, description="Channel, e.g. hotline, sensor, SOC.")

    @field_validator("report", "title", "location", "reported_by", "source")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        """Normalise text fields and reject reports that are empty after cleaning."""
        if value is None:
            return None
        cleaned = clean_text(value)
        return cleaned or None

    @field_validator("report")
    @classmethod
    def _report_has_content(cls, value: str | None) -> str:
        """Ensure the report still contains meaningful text after cleaning."""
        if not value or len(value) < 20:
            raise ValueError("report must contain at least 20 meaningful characters")
        if not re.search(r"[A-Za-z]{3,}", value):
            raise ValueError("report must contain natural-language text")
        return value


# --------------------------------------------------------------------------- #
# Agent outputs (also used as tool-call argument schemas)
# --------------------------------------------------------------------------- #
class IncidentData(BaseModel):
    """Structured incident record produced by the intake agent."""

    title: str = Field(..., min_length=5, max_length=200, description="Concise, factual incident title.")
    description: str = Field(
        ..., min_length=20, max_length=2000, description="Neutral factual summary of what happened."
    )
    category: IncidentCategory = Field(..., description="Best-fit high-level category.")
    location: str | None = Field(default=None, max_length=200, description="Where the incident is happening.")
    reported_at: str | None = Field(default=None, max_length=100, description="Time of the event if stated.")
    affected_assets: list[str] = Field(
        default_factory=list, max_length=20, description="Systems, facilities or infrastructure affected."
    )
    people_affected: int | None = Field(default=None, ge=0, description="Estimated number of people affected.")
    casualties: int | None = Field(default=None, ge=0, description="Reported deaths or injuries, if stated.")
    observed_indicators: list[str] = Field(
        default_factory=list, max_length=20, description="Concrete observed signals or symptoms."
    )
    key_entities: list[str] = Field(
        default_factory=list, max_length=20, description="Named organisations, groups, systems or people."
    )
    extraction_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="0-1 confidence that the extraction is complete and accurate."
    )


class ThreatAssessment(BaseModel):
    """Threat classification produced by the analysis agent's LLM tool call."""

    severity: Severity = Field(..., description="LOW, MEDIUM, HIGH or CRITICAL per the severity rubric.")
    threat_type: str = Field(
        ..., min_length=3, max_length=120, description="Specific threat type, e.g. 'Ransomware', 'Riverine Flood'."
    )
    category: IncidentCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence in this classification.")
    rationale: str = Field(..., min_length=10, max_length=1500, description="Why this severity and type.")
    key_indicators: list[str] = Field(default_factory=list, max_length=15)
    potential_impact: str = Field(..., min_length=5, max_length=1000)
    escalation_required: bool = Field(..., description="Whether executive/leadership escalation is required.")


class AnalysisResult(ThreatAssessment):
    """Threat assessment enriched with deterministic heuristic calibration."""

    heuristic_severity: Severity | None = None
    heuristic_score: float | None = None
    raw_llm_confidence: float | None = None


class RetrievedDocument(BaseModel):
    """A single document returned by semantic search over the knowledge base."""

    doc_id: str
    doc_type: Literal["incident", "protocol"]
    title: str
    content: str
    similarity: float = Field(..., description="Cosine similarity in [−1, 1]; higher is more relevant.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """Aggregated retrieval results produced by the RAG agent."""

    queries: list[str] = Field(default_factory=list)
    past_incidents: list[RetrievedDocument] = Field(default_factory=list)
    protocols: list[RetrievedDocument] = Field(default_factory=list)
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    used_fallback: bool = False

    @property
    def all_documents(self) -> list[RetrievedDocument]:
        """All retrieved documents (incidents first, then protocols)."""
        return [*self.past_incidents, *self.protocols]


class ActionItem(BaseModel):
    """One actionable step of a response strategy."""

    action: str = Field(..., min_length=5, max_length=400, description="Concrete action to take.")
    owner: str = Field(..., min_length=2, max_length=120, description="Responsible team or role.")
    timeframe: str = Field(..., min_length=1, max_length=60, description="When, e.g. '0-1h', 'within 24h'.")
    priority: int = Field(..., ge=1, le=5, description="1 = highest priority, 5 = lowest.")


class ResponseStrategy(BaseModel):
    """Structured, actionable response strategy produced by the response agent."""

    summary: str = Field(..., min_length=20, max_length=1500, description="Executive summary of the strategy.")
    immediate_actions: list[ActionItem] = Field(
        ..., min_length=1, max_length=10, description="Actions for the first 0-4 hours."
    )
    short_term_actions: list[ActionItem] = Field(
        default_factory=list, max_length=10, description="Actions for 4-72 hours."
    )
    long_term_actions: list[ActionItem] = Field(
        default_factory=list, max_length=10, description="Recovery and prevention actions."
    )
    resources_required: list[str] = Field(default_factory=list, max_length=20)
    stakeholders_to_notify: list[str] = Field(default_factory=list, max_length=20)
    communication_plan: str = Field(..., min_length=10, max_length=1500)
    risks_and_mitigations: list[str] = Field(default_factory=list, max_length=15)
    referenced_sources: list[str] = Field(
        default_factory=list, description="IDs of retrieved past incidents/protocols the plan relies on."
    )
    estimated_resolution_time: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence in the strategy.")


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #
class AgentMetrics(BaseModel):
    """Latency, token usage and confidence recorded for one agent execution."""

    agent: str
    status: Literal["success", "error", "skipped"]
    latency_ms: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    confidence: float | None = None
    tool_calls: list[str] = Field(default_factory=list)
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
