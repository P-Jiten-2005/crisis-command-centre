"""Analysis agent: classifies threat severity (LOW/MEDIUM/HIGH/CRITICAL) and threat type."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agents.base import AgentError, AgentOutcome, BaseAgent, make_output_tool
from agents.safeguards import UNTRUSTED_DATA_POLICY, wrap_untrusted
from agents.schemas import AnalysisResult, IncidentCategory, IncidentData, Severity, ThreatAssessment
from agents.state import CrisisState

SEVERITY_RUBRIC: dict[str, str] = {
    "CRITICAL": (
        "Loss of life or imminent threat to life, mass casualties, or failure/compromise of critical "
        "infrastructure (hospitals, power grid, water, 911) affecting large populations; active adversary "
        "in safety-critical systems. Requires immediate executive escalation."
    ),
    "HIGH": (
        "Serious injuries or significant risk to public safety, large-scale service disruption, major data "
        "breach, or rapidly escalating situation that could become critical without swift action."
    ),
    "MEDIUM": (
        "Localized impact, limited disruption, contained threat, or illnesses/injuries without fatalities; "
        "requires coordinated response but not emergency escalation."
    ),
    "LOW": "Minor or potential issue with minimal impact, no injuries, easily contained by routine operations.",
}

THREAT_TAXONOMY: dict[str, list[str]] = {
    IncidentCategory.CYBER_ATTACK.value: [
        "Ransomware", "Distributed Denial of Service", "Data Breach", "Phishing / Credential Compromise",
        "Industrial Control System Intrusion", "Supply Chain Attack", "Insider Threat", "Malware",
    ],
    IncidentCategory.NATURAL_DISASTER.value: [
        "Riverine Flood", "Flash Flood", "Earthquake", "Wildfire", "Hurricane", "Tornado", "Winter Storm",
        "Extreme Heat", "Tsunami", "Landslide",
    ],
    IncidentCategory.INFRASTRUCTURE_FAILURE.value: [
        "Power Grid Failure", "Water System Contamination", "Structural Collapse", "Telecommunications Outage",
        "Gas Pipeline Failure", "Transportation System Failure", "Dam Failure",
    ],
    IncidentCategory.PUBLIC_SAFETY.value: [
        "Active Shooter", "Terrorism", "Hazardous Materials Release", "Crowd Crush", "Public Health Outbreak",
        "Civil Unrest", "Mass Casualty Incident", "Missing Persons",
    ],
    IncidentCategory.OTHER.value: ["Other"],
}


class SeverityHeuristic:
    """Deterministic keyword/impact heuristic used to calibrate LLM severity.

    Keywords ending in ``*`` match as word prefixes; others match whole words.
    """

    INDICATORS: dict[str, tuple[float, tuple[str, ...]]] = {
        "life_threat": (3.0, ("fatalit*", "dead", "death*", "killed", "casualt*", "life-threatening",
                              "critical condition", "trapped", "gunman", "shooter", "hostage*")),
        "critical_infrastructure": (2.0, ("hospital*", "911", "power grid", "blackout", "water treatment", "scada",
                                          "dam", "dams", "nuclear", "air traffic", "emergency services")),
        "hazard_scale": (2.0, ("explosion*", "toxic", "chlorine", "collaps*", "wildfire*", "hurricane*",
                               "earthquake*", "tsunami*", "evacuat*", "plume", "levee*")),
        "cyber_severity": (1.5, ("ransomware", "exfiltrat*", "encrypted", "backdoor*", "lateral movement",
                                 "domain controller*", "breach*")),
        "injury": (1.0, ("injur*", "hospitaliz*", "wounded", "respiratory", "symptom*", "illness*", "sick")),
        "disruption": (0.75, ("outage*", "offline", "disrupt*", "unavailable", "failure*", "failed")),
    }

    def __init__(self) -> None:
        """Pre-compile one regex per indicator group."""
        self._patterns: dict[str, tuple[float, re.Pattern[str]]] = {}
        for group, (weight, keywords) in self.INDICATORS.items():
            parts = [re.escape(kw[:-1]) + r"\w*" if kw.endswith("*") else re.escape(kw) + r"\b" for kw in keywords]
            self._patterns[group] = (weight, re.compile(r"\b(" + "|".join(parts) + ")", re.IGNORECASE))

    def score(self, text: str, people_affected: int | None = None) -> dict[str, Any]:
        """Score free text for severity indicators.

        Args:
            text: Incident text.
            people_affected: Optional estimate of affected people.

        Returns:
            Dict with ``score``, ``severity`` and ``matched_indicators``.
        """
        matched: dict[str, list[str]] = {}
        total = 0.0
        for group, (weight, pattern) in self._patterns.items():
            hits = sorted({m.group(0).lower() for m in pattern.finditer(text)})
            if hits:
                matched[group] = hits
                total += weight
        if people_affected:
            total += 2.0 if people_affected >= 100_000 else 1.0 if people_affected >= 1_000 else 0.0
        numbers = [int(n.replace(",", "")) for n in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b", text)]
        if numbers and max(numbers) >= 100_000 and not people_affected:
            total += 1.0
        severity = (
            Severity.CRITICAL if total >= 5.5 else Severity.HIGH if total >= 3.0 else Severity.MEDIUM if total >= 1.0
            else Severity.LOW
        )
        return {"score": round(total, 2), "severity": severity.value, "matched_indicators": matched}


class _IndicatorInput(BaseModel):
    """Arguments for the ``score_threat_indicators`` tool."""

    text: str = Field(..., description="Incident text to analyse.")


class _TaxonomyInput(BaseModel):
    """Arguments for the ``get_threat_taxonomy`` tool."""

    category: IncidentCategory = Field(..., description="Incident category.")


class AnalysisAgent(BaseAgent):
    """Classifies severity and threat type via tool calling, calibrated by a heuristic."""

    name = "analysis"

    SYSTEM_PROMPT = (
        "You are the Threat Analysis Officer of a Critical Command Crisis Center. Classify the incident's "
        "severity (LOW, MEDIUM, HIGH, CRITICAL) and specific threat type.\n"
        "The severity rubric, the threat taxonomy for the reported category and a deterministic indicator "
        "score are provided below, so normally call `submit_threat_assessment` straight away. Only call "
        "`get_threat_taxonomy` (other categories) or `score_threat_indicators` if you genuinely need more.\n"
        "Prefer threat types from the taxonomy. Be calibrated: `confidence` should be lower when facts are "
        "missing or ambiguous. Set `escalation_required` true for CRITICAL incidents and any HIGH incident "
        "with risk to life.\n"
        f"{UNTRUSTED_DATA_POLICY}"
    )

    def __init__(self, llm: BaseChatModel, heuristic: SeverityHeuristic | None = None, **kwargs: int) -> None:
        """Create the analysis agent.

        Args:
            llm: Tool-calling chat model.
            heuristic: Severity heuristic used for tools and calibration.
            **kwargs: Forwarded to :class:`BaseAgent`.
        """
        super().__init__(llm, **kwargs)
        self.heuristic = heuristic or SeverityHeuristic()
        self.output_tool = make_output_tool(
            "submit_threat_assessment",
            "Submit the final threat severity and type classification.",
            ThreatAssessment,
        )
        self.helper_tools: list[BaseTool] = self._build_helper_tools()

    def _build_helper_tools(self) -> list[BaseTool]:
        """Create the informational LangChain tools available to the model.

        Returns:
            The helper tools.
        """

        def get_severity_rubric() -> dict[str, str]:
            """Return the crisis center's official severity classification rubric."""
            return SEVERITY_RUBRIC

        def get_threat_taxonomy(category: IncidentCategory) -> list[str]:
            """Return the standard threat types for an incident category."""
            key = category.value if isinstance(category, IncidentCategory) else str(category)
            return THREAT_TAXONOMY.get(key, THREAT_TAXONOMY[IncidentCategory.OTHER.value])

        def score_threat_indicators(text: str) -> dict[str, Any]:
            """Run the deterministic keyword-based severity heuristic on incident text."""
            return self.heuristic.score(text)

        return [
            StructuredTool.from_function(get_severity_rubric, name="get_severity_rubric"),
            StructuredTool.from_function(get_threat_taxonomy, name="get_threat_taxonomy", args_schema=_TaxonomyInput),
            StructuredTool.from_function(
                score_threat_indicators, name="score_threat_indicators", args_schema=_IndicatorInput
            ),
        ]

    def run(self, state: CrisisState) -> AgentOutcome:
        """Classify the structured incident.

        Args:
            state: Workflow state containing ``incident_data``.

        Returns:
            Outcome with ``analysis_result`` set.

        Raises:
            AgentError: If intake output is missing.
        """
        if not state.get("incident_data"):
            raise AgentError("incident_data is missing from state")
        incident = IncidentData.model_validate(state["incident_data"])

        raw_report = state.get("raw_report", "")
        heuristic = self.heuristic.score(
            f"{incident.title}\n{incident.description}\n{' '.join(incident.observed_indicators)}\n{raw_report}",
            incident.people_affected,
        )
        # Pre-load reference data so the model can decide in a single LLM call.
        reference = {
            "severity_rubric": SEVERITY_RUBRIC,
            "threat_taxonomy": {incident.category.value: THREAT_TAXONOMY[incident.category.value]},
            "indicator_score": heuristic,
        }
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Structured incident record:\n"
                    f"{self.to_json(incident.model_dump(mode='json'))}\n\n"
                    f"Reference data:\n{self.to_json(reference)}\n\n"
                    "Original report for reference:\n"
                    f"{wrap_untrusted(raw_report, 'incident_report')}"
                )
            ),
        ]
        assessment = self.run_tool_loop(messages, self.output_tool, ThreatAssessment, self.helper_tools)
        result = self._calibrate(assessment, heuristic)
        return AgentOutcome(
            update={"analysis_result": result.model_dump(mode="json")},
            confidence=result.confidence,
            memory_note={
                "severity": result.severity.value,
                "threat_type": result.threat_type,
                "heuristic_severity": result.heuristic_severity.value if result.heuristic_severity else None,
            },
        )

    @staticmethod
    def _calibrate(assessment: ThreatAssessment, heuristic: dict[str, Any]) -> AnalysisResult:
        """Adjust confidence based on agreement with the deterministic heuristic.

        Args:
            assessment: LLM assessment.
            heuristic: Output of :meth:`SeverityHeuristic.score` for this incident.

        Returns:
            The calibrated :class:`AnalysisResult`.
        """
        heuristic_severity = Severity(heuristic["severity"])
        gap = abs(assessment.severity.rank - heuristic_severity.rank)
        factor = {0: 1.0, 1: 0.9}.get(gap, 0.7)
        escalate = assessment.escalation_required or assessment.severity == Severity.CRITICAL
        return AnalysisResult(
            **assessment.model_dump(exclude={"confidence", "escalation_required"}),
            confidence=round(assessment.confidence * factor, 4),
            escalation_required=escalate,
            heuristic_severity=heuristic_severity,
            heuristic_score=heuristic["score"],
            raw_llm_confidence=assessment.confidence,
        )
