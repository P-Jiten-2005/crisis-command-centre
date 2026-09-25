"""Intake agent: parses raw incident reports into a structured ``IncidentData`` model."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import AgentError, AgentOutcome, BaseAgent, make_output_tool
from agents.safeguards import UNTRUSTED_DATA_POLICY, wrap_untrusted
from agents.schemas import IncidentData
from agents.state import CrisisState


class IntakeAgent(BaseAgent):
    """Extracts a validated, structured incident record from free text via tool calling."""

    name = "intake"

    SYSTEM_PROMPT = (
        "You are the Intake Officer of a Critical Command Crisis Center. Convert the raw incident report into "
        "a structured incident record by calling the `submit_incident_record` tool exactly once.\n"
        "Rules:\n"
        "- Extract only facts stated in or directly implied by the report. Do not invent numbers, names or places; "
        "leave unknown optional fields empty or null.\n"
        "- `description` is a neutral, factual 2-5 sentence summary.\n"
        "- Choose the best-fit `category`: CYBER_ATTACK, NATURAL_DISASTER, INFRASTRUCTURE_FAILURE, "
        "PUBLIC_SAFETY or OTHER.\n"
        "- `extraction_confidence` reflects how complete and unambiguous the report is (vague reports < 0.6).\n"
        f"{UNTRUSTED_DATA_POLICY}"
    )

    def __init__(self, llm: BaseChatModel, **kwargs: int) -> None:
        """Create the intake agent.

        Args:
            llm: Tool-calling chat model.
            **kwargs: Forwarded to :class:`BaseAgent`.
        """
        super().__init__(llm, **kwargs)
        self.output_tool = make_output_tool(
            "submit_incident_record",
            "Submit the structured incident record extracted from the raw report.",
            IncidentData,
        )

    def run(self, state: CrisisState) -> AgentOutcome:
        """Parse the raw report into :class:`IncidentData`.

        Args:
            state: Workflow state containing ``raw_report`` and ``submission``.

        Returns:
            Outcome with ``incident_data`` set.

        Raises:
            AgentError: If the report is missing.
        """
        report = state.get("raw_report")
        if not report:
            raise AgentError("raw_report is missing from state")
        submission = state.get("submission") or {}
        hints = {k: v for k, v in submission.items() if k in {"title", "location", "reported_by", "source"} and v}

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Submission metadata (may be empty): {self.to_json(hints)}\n\n"
                    f"{wrap_untrusted(report, 'incident_report')}"
                )
            ),
        ]
        data = self.run_tool_loop(messages, self.output_tool, IncidentData)

        # Operator-provided metadata takes precedence over model extraction.
        overrides = {k: submission[k] for k in ("title", "location") if submission.get(k)}
        if overrides:
            data = data.model_copy(update=overrides)

        return AgentOutcome(
            update={"incident_data": data.model_dump(mode="json")},
            confidence=data.extraction_confidence,
            memory_note={"category": data.category.value, "title": data.title},
        )
