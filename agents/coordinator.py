"""Coordinator: LangGraph ``StateGraph`` orchestrating all agents with shared state.

Graph topology::

    START -> guardrail -> intake -> analysis -> rag -> response -> finalize -> END
                 |           |          |
                 +-----------+----------+--> finalize   (on block / failure)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.analysis_agent import AnalysisAgent
from agents.intake_agent import IntakeAgent
from agents.rag_agent import RAGAgent
from agents.response_agent import ResponseAgent
from agents.safeguards import PromptInjectionDetector
from agents.schemas import AgentMetrics
from agents.state import CrisisState

logger = logging.getLogger(__name__)


class WorkflowStatus:
    """String constants for workflow status values."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class CrisisCoordinator:
    """Builds and runs the multi-agent crisis workflow graph."""

    #: Weights used to combine per-agent confidences into an overall score.
    CONFIDENCE_WEIGHTS: dict[str, float] = {"intake": 0.15, "analysis": 0.35, "rag": 0.2, "response": 0.3}

    def __init__(
        self,
        intake: IntakeAgent,
        analysis: AnalysisAgent,
        rag: RAGAgent,
        response: ResponseAgent,
        detector: PromptInjectionDetector | None = None,
    ) -> None:
        """Wire the agents into a compiled graph.

        Args:
            intake: Intake agent node.
            analysis: Analysis agent node.
            rag: RAG agent node.
            response: Response agent node.
            detector: Prompt-injection detector for the guardrail node.
        """
        self.intake = intake
        self.analysis = analysis
        self.rag = rag
        self.response = response
        self.detector = detector or PromptInjectionDetector()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self) -> Any:
        """Assemble and compile the :class:`StateGraph`.

        Returns:
            The compiled graph.
        """
        graph = StateGraph(CrisisState)
        graph.add_node("guardrail", self._guardrail_node)
        graph.add_node("intake", self.intake)
        graph.add_node("analysis", self.analysis)
        graph.add_node("rag", self.rag)
        graph.add_node("response", self.response)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "guardrail")
        graph.add_conditional_edges(
            "guardrail", self._route_after_guardrail, {"intake": "intake", "finalize": "finalize"}
        )
        graph.add_conditional_edges(
            "intake", self._route_on("incident_data", "analysis"), {"analysis": "analysis", "finalize": "finalize"}
        )
        graph.add_conditional_edges(
            "analysis", self._route_on("analysis_result", "rag"), {"rag": "rag", "finalize": "finalize"}
        )
        graph.add_edge("rag", "response")  # RAG degrades gracefully, so always continue
        graph.add_edge("response", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    @staticmethod
    def _route_after_guardrail(state: CrisisState) -> str:
        """Route to intake unless the input was blocked.

        Args:
            state: Workflow state.

        Returns:
            Next node name.
        """
        return "finalize" if state.get("status") == WorkflowStatus.BLOCKED else "intake"

    @staticmethod
    def _route_on(required_key: str, next_node: str) -> Any:
        """Build a router that continues only when ``required_key`` is present in state.

        Args:
            required_key: State key produced by the previous node.
            next_node: Node to route to on success.

        Returns:
            A routing function.
        """

        def _router(state: CrisisState) -> str:
            """Return ``next_node`` if the previous agent succeeded, else ``finalize``."""
            return next_node if state.get(required_key) else "finalize"

        return _router

    # ------------------------------------------------------------------ #
    # Non-agent nodes
    # ------------------------------------------------------------------ #
    def _guardrail_node(self, state: CrisisState) -> dict[str, Any]:
        """Screen the raw report for prompt injection before any LLM sees it.

        Args:
            state: Workflow state.

        Returns:
            State update with sanitised report or a BLOCKED status.
        """
        result = self.detector.check(state.get("raw_report", ""))
        guard_memory = {"guardrail": {"risk_score": result.risk_score, "matched_rules": result.matched_rules}}
        if result.is_injection:
            logger.warning("guardrail: blocked report (score=%.2f rules=%s)", result.risk_score, result.matched_rules)
            return {
                "status": WorkflowStatus.BLOCKED,
                "errors": [f"guardrail: prompt injection detected ({', '.join(result.matched_rules)})"],
                "memory": guard_memory,
            }
        return {"raw_report": result.sanitized_text, "status": WorkflowStatus.RUNNING, "memory": guard_memory}

    def _finalize_node(self, state: CrisisState) -> dict[str, Any]:
        """Compute final status, overall confidence and aggregate metrics.

        Args:
            state: Workflow state.

        Returns:
            Final state update.
        """
        metrics = [AgentMetrics.model_validate(m) for m in state.get("metrics", [])]
        if state.get("status") == WorkflowStatus.BLOCKED:
            status = WorkflowStatus.BLOCKED
        elif state.get("response_strategy"):
            status = WorkflowStatus.COMPLETED
        else:
            status = WorkflowStatus.FAILED

        weighted, weight_sum = 0.0, 0.0
        for m in metrics:
            weight = self.CONFIDENCE_WEIGHTS.get(m.agent, 0.0)
            if m.confidence is not None and weight:
                weighted += m.confidence * weight
                weight_sum += weight
        overall = round(weighted / weight_sum, 4) if weight_sum and status == WorkflowStatus.COMPLETED else None

        summary = {
            "total_latency_ms": round(sum(m.latency_ms for m in metrics), 2),
            "total_tokens": sum(m.total_tokens for m in metrics),
            "input_tokens": sum(m.input_tokens for m in metrics),
            "output_tokens": sum(m.output_tokens for m in metrics),
            "llm_calls": sum(m.llm_calls for m in metrics),
            "agents": {m.agent: {"latency_ms": m.latency_ms, "tokens": m.total_tokens, "status": m.status}
                       for m in metrics},
        }
        logger.info(
            "workflow status=%s overall_confidence=%s latency_ms=%.0f tokens=%d",
            status, overall, summary["total_latency_ms"], summary["total_tokens"],
        )
        return {"status": status, "overall_confidence": overall, "memory": {"run_summary": summary}}

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(
        self,
        raw_report: str,
        submission: dict[str, Any] | None = None,
        incident_id: str | None = None,
        memory: dict[str, Any] | None = None,
    ) -> CrisisState:
        """Execute the full workflow synchronously.

        Args:
            raw_report: The raw incident report text.
            submission: Submission metadata (title, location, ...).
            incident_id: Database ID for tracing.
            memory: Initial memory (e.g. recent incidents loaded from PostgreSQL).

        Returns:
            The final workflow state.
        """
        initial, config = self._prepare(raw_report, submission, incident_id, memory)
        return self.graph.invoke(initial, config=config)

    def stream(
        self,
        raw_report: str,
        submission: dict[str, Any] | None = None,
        incident_id: str | None = None,
        memory: dict[str, Any] | None = None,
    ) -> Iterator[CrisisState]:
        """Execute the workflow, yielding the full state after every node.

        Args:
            raw_report: The raw incident report text.
            submission: Submission metadata (title, location, ...).
            incident_id: Database ID for tracing.
            memory: Initial memory (e.g. recent incidents loaded from PostgreSQL).

        Yields:
            The accumulated workflow state after each step; the last one is final.
        """
        initial, config = self._prepare(raw_report, submission, incident_id, memory)
        yield from self.graph.stream(initial, config=config, stream_mode="values")

    @staticmethod
    def _prepare(
        raw_report: str,
        submission: dict[str, Any] | None,
        incident_id: str | None,
        memory: dict[str, Any] | None,
    ) -> tuple[CrisisState, dict[str, Any]]:
        """Build the initial state and run config.

        Args:
            raw_report: The raw incident report text.
            submission: Submission metadata.
            incident_id: Database ID for tracing.
            memory: Initial memory.

        Returns:
            The initial state and the LangGraph run config.
        """
        initial: CrisisState = {
            "incident_id": incident_id or "",
            "raw_report": raw_report,
            "submission": submission or {},
            "incident_data": None,
            "analysis_result": None,
            "retrieved_context": None,
            "response_strategy": None,
            "memory": memory or {},
            "metrics": [],
            "errors": [],
            "status": WorkflowStatus.RUNNING,
            "overall_confidence": None,
        }
        config = {"run_name": "crisis-workflow", "tags": [f"incident:{incident_id}"], "recursion_limit": 25}
        return initial, config

    def mermaid(self) -> str:
        """Return a Mermaid diagram of the compiled graph.

        Returns:
            Mermaid source.
        """
        return self.graph.get_graph().draw_mermaid()
