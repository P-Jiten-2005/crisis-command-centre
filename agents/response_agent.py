"""Response agent: generates a structured, actionable response strategy grounded in retrieved context."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import AgentError, AgentOutcome, BaseAgent, make_output_tool
from agents.safeguards import UNTRUSTED_DATA_POLICY, wrap_untrusted
from agents.schemas import AnalysisResult, IncidentData, ResponseStrategy, RetrievedContext, Severity
from agents.state import CrisisState


class ResponseAgent(BaseAgent):
    """Produces a validated :class:`ResponseStrategy` via a forced tool call."""

    name = "response"

    SYSTEM_PROMPT = (
        "You are the Response Strategy Commander of a Critical Command Crisis Center. Produce an actionable "
        "response strategy by calling the `submit_response_strategy` tool exactly once.\n"
        "Requirements:\n"
        "- Ground the plan in the retrieved past incidents and protocols; adapt their lessons learned. List "
        "the IDs you relied on in `referenced_sources` (only IDs that appear in the retrieved documents).\n"
        "- Actions must be concrete and assignable: each has an owner (team/role), a timeframe and priority "
        "1 (highest) to 5.\n"
        "- Immediate actions cover 0-4 hours and put life safety first; then containment, then recovery.\n"
        "- Scale the response to the severity. For HIGH/CRITICAL include leadership and regulatory "
        "stakeholders in `stakeholders_to_notify`.\n"
        "- Consider related ongoing incidents from memory if relevant (possible coordinated or cascading events).\n"
        "- `confidence` is lower when retrieved context is weak or facts are missing.\n"
        f"{UNTRUSTED_DATA_POLICY}"
    )

    def __init__(self, llm: BaseChatModel, **kwargs: int) -> None:
        """Create the response agent.

        Args:
            llm: Tool-calling chat model.
            **kwargs: Forwarded to :class:`BaseAgent`.
        """
        super().__init__(llm, **kwargs)
        self.output_tool = make_output_tool(
            "submit_response_strategy",
            "Submit the final structured crisis response strategy.",
            ResponseStrategy,
        )

    def run(self, state: CrisisState) -> AgentOutcome:
        """Generate the response strategy.

        Args:
            state: Workflow state with intake, analysis and retrieval outputs.

        Returns:
            Outcome with ``response_strategy`` set.

        Raises:
            AgentError: If upstream outputs are missing.
        """
        if not state.get("incident_data") or not state.get("analysis_result"):
            raise AgentError("incident_data and analysis_result are required")
        incident = IncidentData.model_validate(state["incident_data"])
        analysis = AnalysisResult.model_validate(state["analysis_result"])
        context = RetrievedContext.model_validate(state.get("retrieved_context") or {})
        recent = (state.get("memory") or {}).get("recent_incidents", [])

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=self._build_prompt(incident, analysis, context, recent)),
        ]
        strategy = self.run_tool_loop(messages, self.output_tool, ResponseStrategy)
        strategy = self._post_process(strategy, analysis, context)
        return AgentOutcome(
            update={"response_strategy": strategy.model_dump(mode="json")},
            confidence=strategy.confidence,
            memory_note={
                "immediate_actions": len(strategy.immediate_actions),
                "referenced_sources": strategy.referenced_sources,
            },
        )

    def _build_prompt(
        self,
        incident: IncidentData,
        analysis: AnalysisResult,
        context: RetrievedContext,
        recent: list[dict[str, Any]],
    ) -> str:
        """Assemble the user prompt with clearly delimited untrusted sections.

        Args:
            incident: Structured incident.
            analysis: Threat analysis.
            context: Retrieved knowledge.
            recent: Recent incidents from long-term memory.

        Returns:
            Prompt text.
        """
        docs = "\n".join(
            wrap_untrusted(doc.content, "retrieved_document", f'id="{doc.doc_id}" type="{doc.doc_type}" '
                           f'similarity="{doc.similarity:.2f}"')
            for doc in context.all_documents
        ) or "(no relevant documents retrieved - rely on general best practice and lower your confidence)"
        related = "\n".join(wrap_untrusted(self.to_json(item), "prior_incident") for item in recent) or "(none)"
        analysis_view = analysis.model_dump(mode="json", exclude={"raw_llm_confidence", "heuristic_score"})
        return (
            f"INCIDENT:\n{self.to_json(incident.model_dump(mode='json'))}\n\n"
            f"THREAT ANALYSIS:\n{self.to_json(analysis_view)}\n\n"
            f"RETRIEVED KNOWLEDGE:\n{docs}\n\n"
            f"RECENT INCIDENTS FROM MEMORY:\n{related}"
        )

    @staticmethod
    def _post_process(
        strategy: ResponseStrategy, analysis: AnalysisResult, context: RetrievedContext
    ) -> ResponseStrategy:
        """Enforce grounding and escalation rules and calibrate confidence.

        Args:
            strategy: Raw validated strategy from the model.
            analysis: Threat analysis.
            context: Retrieved knowledge.

        Returns:
            The adjusted strategy.
        """
        valid_ids = {d.doc_id for d in context.all_documents}
        references = [ref for ref in dict.fromkeys(strategy.referenced_sources) if ref in valid_ids]

        stakeholders = list(strategy.stakeholders_to_notify)
        if analysis.escalation_required and not any(
            word in s.lower() for s in stakeholders for word in ("executive", "leadership", "director", "command")
        ):
            stakeholders.insert(0, "Executive crisis leadership")

        retrieval = context.retrieval_confidence
        confidence = 0.6 * strategy.confidence + 0.4 * retrieval if context.all_documents else 0.8 * strategy.confidence
        if analysis.severity == Severity.CRITICAL and not references:
            confidence *= 0.85  # ungrounded plan for a critical incident

        immediate = sorted(strategy.immediate_actions, key=lambda a: a.priority)
        return strategy.model_copy(
            update={
                "referenced_sources": references,
                "stakeholders_to_notify": stakeholders[:20],
                "immediate_actions": immediate,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
            }
        )
