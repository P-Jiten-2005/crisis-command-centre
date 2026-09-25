"""Pydantic response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentRunOut(BaseModel):
    """Per-agent execution metrics."""

    model_config = ConfigDict(from_attributes=True)

    agent_name: str
    status: str
    latency_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    confidence: float | None = None
    tool_calls: list[str] | None = None
    error: str | None = None
    created_at: datetime | None = None


class IncidentSummary(BaseModel):
    """Compact incident view used in listings."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    severity: str | None = None
    threat_type: str | None = None
    category: str | None = None
    status: str
    confidence: float | None = None
    created_at: datetime | None = None


class IncidentDetail(IncidentSummary):
    """Full incident view including every agent's output and metrics."""

    description: str
    raw_report: str
    location: str | None = None
    reported_by: str | None = None
    source: str | None = None
    incident_data: dict[str, Any] | None = None
    analysis_result: dict[str, Any] | None = None
    retrieved_context: dict[str, Any] | None = None
    response_strategy: dict[str, Any] | None = None
    workflow_memory: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    updated_at: datetime | None = None
    agent_runs: list[AgentRunOut] = []


class IncidentList(BaseModel):
    """Paginated list of incidents."""

    total: int
    skip: int
    limit: int
    items: list[IncidentSummary]


class InjectionRejection(BaseModel):
    """Error body returned when a report is rejected by the prompt-injection guard."""

    detail: str
    risk_score: float
    matched_rules: list[str]


class HealthStatus(BaseModel):
    """Service health information."""

    status: str
    database: str
    knowledge_base: dict[str, int] | str
    model: str
