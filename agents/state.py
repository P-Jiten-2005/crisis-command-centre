"""Shared LangGraph state carried across all agent nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_memory(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """LangGraph reducer that shallow-merges memory updates from each node.

    Args:
        left: Existing memory.
        right: Update produced by a node.

    Returns:
        The merged memory dict.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class CrisisState(TypedDict, total=False):
    """State passed between nodes of the crisis workflow graph.

    Attributes:
        incident_id: Database ID of the incident being processed.
        raw_report: Sanitised raw report text.
        submission: Original submission metadata (title, location, ...).
        incident_data: Output of the intake agent (``IncidentData`` JSON).
        analysis_result: Output of the analysis agent (``AnalysisResult`` JSON).
        retrieved_context: Output of the RAG agent (``RetrievedContext`` JSON).
        response_strategy: Output of the response agent (``ResponseStrategy`` JSON).
        memory: Shared working memory; each node adds its own notes, and the workflow
            preloads long-term memory (recent incidents) from PostgreSQL.
        metrics: Per-agent ``AgentMetrics`` JSON, appended by each node.
        errors: Error messages appended by any node.
        status: Workflow status (RUNNING, COMPLETED, FAILED, BLOCKED).
        overall_confidence: Weighted confidence across agents, set at the end.
    """

    incident_id: str
    raw_report: str
    submission: dict[str, Any]
    incident_data: dict[str, Any] | None
    analysis_result: dict[str, Any] | None
    retrieved_context: dict[str, Any] | None
    response_strategy: dict[str, Any] | None
    memory: Annotated[dict[str, Any], merge_memory]
    metrics: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    status: str
    overall_confidence: float | None
