"""SQLAlchemy ORM models for persistent incident logging and agent memory."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class IncidentStatus(str, enum.Enum):
    """Lifecycle states of an incident as it moves through the workflow."""

    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


def _new_id() -> str:
    """Generate a new UUID4 primary key.

    Returns:
        The UUID as a string.
    """
    return str(uuid.uuid4())


class Incident(Base):
    """A crisis incident and the full multi-agent analysis produced for it."""

    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str | None] = mapped_column(String(16), index=True)
    threat_type: Mapped[str | None] = mapped_column(String(128), index=True)
    response_strategy: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(16), index=True, default=IncidentStatus.RECEIVED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Extended fields for traceability and agent memory.
    raw_report: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), index=True)
    location: Mapped[str | None] = mapped_column(String(255))
    reported_by: Mapped[str | None] = mapped_column(String(120))
    source: Mapped[str | None] = mapped_column(String(60))
    incident_data: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    analysis_result: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    retrieved_context: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    workflow_memory: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    confidence: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        order_by="AgentRun.id",
        lazy="selectin",
    )

    @property
    def total_latency_ms(self) -> float | None:
        """End-to-end agent latency for this incident, if processing has finished."""
        return (self.metrics or {}).get("total_latency_ms")

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return f"<Incident id={self.id} severity={self.severity} status={self.status}>"


class AgentRun(Base):
    """Per-agent execution record: latency, token usage and confidence."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[float | None] = mapped_column(Float)
    tool_calls: Mapped[list[str] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    incident: Mapped[Incident] = relationship(back_populates="agent_runs")

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return f"<AgentRun agent={self.agent_name} status={self.status} latency_ms={self.latency_ms:.0f}>"
