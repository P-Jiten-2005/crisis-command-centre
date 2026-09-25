"""Workflow service: wires agents into the coordinator and persists results to PostgreSQL."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.analysis_agent import AnalysisAgent
from agents.base import build_llm
from agents.coordinator import CrisisCoordinator, WorkflowStatus
from agents.intake_agent import IntakeAgent
from agents.rag_agent import RAGAgent, SearchableKnowledgeBase
from agents.response_agent import ResponseAgent
from agents.safeguards import PromptInjectionDetector
from agents.schemas import AgentMetrics, IncidentSubmission
from agents.state import CrisisState
from config import Settings, get_settings
from db.database import Database, get_database
from db.models import AgentRun, Incident, IncidentStatus

logger = logging.getLogger(__name__)


class CrisisWorkflow:
    """High-level service: create incidents, run the agent graph and store every result."""

    def __init__(
        self,
        coordinator: CrisisCoordinator,
        database: Database,
        knowledge_base: SearchableKnowledgeBase | None = None,
        memory_window: int = 5,
    ) -> None:
        """Create the workflow service.

        Args:
            coordinator: Compiled agent coordinator.
            database: Database wrapper used for persistence.
            knowledge_base: Knowledge base (exposed for health checks/ingestion).
            memory_window: Number of recent incidents loaded as long-term memory.
        """
        self.coordinator = coordinator
        self.database = database
        self.knowledge_base = knowledge_base
        self.memory_window = memory_window

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CrisisWorkflow":
        """Build the full production stack (Groq LLM, ChromaDB, PostgreSQL) from settings.

        Args:
            settings: Application settings (defaults to environment).

        Returns:
            A ready-to-use workflow.
        """
        from knowledge_base.vector_store import get_knowledge_base

        settings = settings or get_settings()
        llm = build_llm(settings)
        kb = get_knowledge_base()
        detector = PromptInjectionDetector(settings.injection_threshold, settings.max_report_chars)
        coordinator = CrisisCoordinator(
            intake=IntakeAgent(llm),
            analysis=AnalysisAgent(llm),
            rag=RAGAgent(
                llm,
                kb,
                top_k_incidents=settings.rag_top_k_incidents,
                top_k_protocols=settings.rag_top_k_protocols,
                min_similarity=settings.rag_min_similarity,
                detector=detector,
            ),
            response=ResponseAgent(llm),
            detector=detector,
        )
        return cls(coordinator, get_database(), kb, settings.memory_window)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def create_incident(self, session: Session, submission: IncidentSubmission) -> Incident:
        """Persist a newly received incident before processing.

        Args:
            session: Open database session.
            submission: Validated submission.

        Returns:
            The persisted :class:`Incident` with status RECEIVED.
        """
        incident = Incident(
            title=submission.title or self._provisional_title(submission.report),
            description=submission.report,
            raw_report=submission.report,
            location=submission.location,
            reported_by=submission.reported_by,
            source=submission.source,
            status=IncidentStatus.RECEIVED.value,
            workflow_memory={"submission": submission.model_dump(exclude={"report"})},
        )
        session.add(incident)
        session.commit()
        session.refresh(incident)
        logger.info("Incident %s received", incident.id)
        return incident

    def process_incident(self, incident_id: str) -> None:
        """Run the agent graph for a stored incident and persist all outputs.

        Uses its own session so it can run in a background task.

        Args:
            incident_id: ID of the incident to process.
        """
        with self.database.session_scope() as session:
            incident = session.get(Incident, incident_id)
            if incident is None:
                logger.error("Incident %s not found", incident_id)
                return
            incident.status = IncidentStatus.PROCESSING.value
            submission = dict((incident.workflow_memory or {}).get("submission") or {})
            raw_report = incident.raw_report
            memory = self._load_memory(session, exclude_id=incident_id)

        state: CrisisState | None = None
        recorded = 0
        try:
            # Stream node-by-node so each agent's metrics are stored as soon as it finishes;
            # clients polling GET /incidents/{id} see the pipeline progress live.
            for state in self.coordinator.stream(raw_report, submission, incident_id, memory):
                metrics = state.get("metrics") or []
                if len(metrics) > recorded:
                    self._record_agent_runs(incident_id, metrics[recorded:])
                    recorded = len(metrics)
            if state is None:
                raise RuntimeError("workflow produced no state")
        except Exception as exc:  # noqa: BLE001 - persist failure instead of losing the incident
            logger.exception("Workflow crashed for incident %s", incident_id)
            with self.database.session_scope() as session:
                incident = session.get(Incident, incident_id)
                if incident is not None:
                    incident.status = IncidentStatus.FAILED.value
                    incident.error = f"{type(exc).__name__}: {exc}"
            return

        with self.database.session_scope() as session:
            incident = session.get(Incident, incident_id)
            if incident is not None:
                self._persist_state(incident, state)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_memory(self, session: Session, exclude_id: str) -> dict[str, Any]:
        """Load long-term memory: summaries of the most recent processed incidents.

        Args:
            session: Open database session.
            exclude_id: The incident currently being processed.

        Returns:
            Initial memory dict for the graph.
        """
        if self.memory_window <= 0:
            return {"recent_incidents": []}
        rows = session.scalars(
            select(Incident)
            .where(Incident.id != exclude_id, Incident.status == IncidentStatus.COMPLETED.value)
            .order_by(Incident.created_at.desc())
            .limit(self.memory_window)
        ).all()
        recent = [
            {
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "severity": r.severity,
                "threat_type": r.threat_type,
                "location": r.location,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        return {"recent_incidents": recent}

    @staticmethod
    def _persist_state(incident: Incident, state: CrisisState) -> None:
        """Copy the final graph state onto the ORM incident (agent runs are stored while streaming).

        Args:
            incident: ORM incident to update.
            state: Final workflow state.
        """
        data = state.get("incident_data") or {}
        analysis = state.get("analysis_result") or {}
        status = state.get("status", WorkflowStatus.FAILED)

        if data:
            incident.title = data.get("title") or incident.title
            incident.description = data.get("description") or incident.description
            incident.category = data.get("category")
            incident.location = incident.location or data.get("location")
        if analysis:
            incident.severity = analysis.get("severity")
            incident.threat_type = analysis.get("threat_type")
            incident.category = analysis.get("category") or incident.category

        incident.incident_data = state.get("incident_data")
        incident.analysis_result = state.get("analysis_result")
        incident.retrieved_context = state.get("retrieved_context")
        incident.response_strategy = state.get("response_strategy")
        incident.confidence = state.get("overall_confidence")
        memory = state.get("memory") or {}
        incident.workflow_memory = {
            "submission": state.get("submission") or {},
            **{k: v for k, v in memory.items() if k != "run_summary"},
        }
        incident.metrics = memory.get("run_summary")
        incident.status = {
            WorkflowStatus.COMPLETED: IncidentStatus.COMPLETED,
            WorkflowStatus.BLOCKED: IncidentStatus.BLOCKED,
        }.get(status, IncidentStatus.FAILED).value
        errors = state.get("errors") or []
        incident.error = "; ".join(errors) if errors else None
        logger.info("Incident %s stored with status %s", incident.id, incident.status)

    def _record_agent_runs(self, incident_id: str, metrics: list[dict[str, Any]]) -> None:
        """Persist per-agent metrics as soon as the agents finish (live progress).

        Args:
            incident_id: Incident being processed.
            metrics: New ``AgentMetrics`` JSON entries to store.
        """
        with self.database.session_scope() as session:
            incident = session.get(Incident, incident_id)
            if incident is None:
                return
            for raw in metrics:
                m = AgentMetrics.model_validate(raw)
                incident.agent_runs.append(
                    AgentRun(
                        agent_name=m.agent,
                        status=m.status,
                        latency_ms=m.latency_ms,
                        llm_calls=m.llm_calls,
                        input_tokens=m.input_tokens,
                        output_tokens=m.output_tokens,
                        total_tokens=m.total_tokens,
                        confidence=m.confidence,
                        tool_calls=m.tool_calls,
                        error=m.error,
                    )
                )

    @staticmethod
    def _provisional_title(report: str) -> str:
        """Derive a placeholder title from the report until intake produces one.

        Args:
            report: Raw report text.

        Returns:
            A title of at most 80 characters.
        """
        first_line = report.strip().splitlines()[0] if report.strip() else "Untitled incident"
        return first_line[:77] + "..." if len(first_line) > 80 else first_line


@lru_cache(maxsize=1)
def get_workflow() -> CrisisWorkflow:
    """Return the process-wide workflow built from environment settings.

    Returns:
        The singleton :class:`CrisisWorkflow`.
    """
    return CrisisWorkflow.from_settings()
