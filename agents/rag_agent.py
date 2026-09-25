"""RAG agent: retrieves relevant past incidents and response protocols from ChromaDB."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agents.base import AgentError, AgentOutcome, BaseAgent
from agents.safeguards import PromptInjectionDetector
from agents.schemas import AnalysisResult, IncidentCategory, IncidentData, RetrievedContext, RetrievedDocument
from agents.state import CrisisState

logger = logging.getLogger(__name__)


class SearchableKnowledgeBase(Protocol):
    """Structural interface the RAG agent needs from the knowledge base."""

    def search_incidents(self, query: str, k: int = 4, category: str | None = None) -> list[RetrievedDocument]:
        """Search past incidents."""

    def search_protocols(self, query: str, k: int = 3, category: str | None = None) -> list[RetrievedDocument]:
        """Search response protocols."""


class _SearchInput(BaseModel):
    """Arguments for the knowledge-base search tools."""

    query: str = Field(..., min_length=3, max_length=300, description="Focused natural-language search query.")
    category: IncidentCategory | None = Field(default=None, description="Optional category filter.")
    k: int = Field(default=3, ge=1, le=5, description="Number of results to return.")


class RAGAgent(BaseAgent):
    """Plans semantic searches with tool calling, executes them and ranks the results."""

    name = "rag"

    SYSTEM_PROMPT = (
        "You are the Knowledge Retrieval Officer of a Critical Command Crisis Center. Plan semantic searches "
        "over the knowledge base to find (a) similar past incidents and (b) applicable response protocols.\n"
        "Call `search_past_incidents` and `search_response_protocols` (you may call each up to twice, in "
        "parallel) with short, focused queries built from the threat type, affected assets and key "
        "indicators. Use the category filter when confident about the category."
    )

    MAX_DOC_CHARS = 1100

    def __init__(
        self,
        llm: BaseChatModel,
        knowledge_base: SearchableKnowledgeBase,
        top_k_incidents: int = 3,
        top_k_protocols: int = 2,
        min_similarity: float = 0.2,
        detector: PromptInjectionDetector | None = None,
        **kwargs: int,
    ) -> None:
        """Create the RAG agent.

        Args:
            llm: Tool-calling chat model.
            knowledge_base: Vector store to search.
            top_k_incidents: Maximum past incidents kept in the context.
            top_k_protocols: Maximum protocols kept in the context.
            min_similarity: Minimum cosine similarity for a document to be kept.
            detector: Injection detector used to screen retrieved content.
            **kwargs: Forwarded to :class:`BaseAgent`.
        """
        super().__init__(llm, **kwargs)
        self.kb = knowledge_base
        self.top_k_incidents = top_k_incidents
        self.top_k_protocols = top_k_protocols
        self.min_similarity = min_similarity
        self.detector = detector or PromptInjectionDetector()
        self.tools: list[BaseTool] = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        """Create LangChain tools wrapping the knowledge base.

        Returns:
            The search tools.
        """

        def search_past_incidents(query: str, category: IncidentCategory | None = None, k: int = 3) -> list[dict]:
            """Semantic search over historical crisis incidents and how they were handled."""
            cat = category.value if isinstance(category, IncidentCategory) else category
            return [d.model_dump() for d in self.kb.search_incidents(query, k=k, category=cat)]

        def search_response_protocols(query: str, category: IncidentCategory | None = None, k: int = 3) -> list[dict]:
            """Semantic search over standard crisis response protocols and playbooks."""
            cat = category.value if isinstance(category, IncidentCategory) else category
            return [d.model_dump() for d in self.kb.search_protocols(query, k=k, category=cat)]

        return [
            StructuredTool.from_function(search_past_incidents, name="search_past_incidents", args_schema=_SearchInput),
            StructuredTool.from_function(
                search_response_protocols, name="search_response_protocols", args_schema=_SearchInput
            ),
        ]

    def run(self, state: CrisisState) -> AgentOutcome:
        """Retrieve context for the current incident.

        Args:
            state: Workflow state containing ``incident_data`` and ``analysis_result``.

        Returns:
            Outcome with ``retrieved_context`` set.

        Raises:
            AgentError: If upstream outputs are missing.
        """
        if not state.get("incident_data") or not state.get("analysis_result"):
            raise AgentError("incident_data and analysis_result are required")
        incident = IncidentData.model_validate(state["incident_data"])
        analysis = AnalysisResult.model_validate(state["analysis_result"])

        queries: list[str] = []
        incidents: dict[str, RetrievedDocument] = {}
        protocols: dict[str, RetrievedDocument] = {}
        used_fallback = False

        try:
            self._llm_planned_search(incident, analysis, queries, incidents, protocols)
        except AgentError as exc:
            logger.warning("rag: LLM query planning failed, using fallback queries: %s", exc)

        if not incidents or not protocols:
            used_fallback = True
            self._fallback_search(incident, analysis, queries, incidents, protocols)

        context = RetrievedContext(
            queries=queries,
            past_incidents=self._rank(incidents, self.top_k_incidents),
            protocols=self._rank(protocols, self.top_k_protocols),
            used_fallback=used_fallback,
        )
        context.retrieval_confidence = self._confidence(context)
        return AgentOutcome(
            update={"retrieved_context": context.model_dump(mode="json")},
            confidence=context.retrieval_confidence,
            memory_note={
                "queries": queries,
                "sources": [d.doc_id for d in context.all_documents],
                "used_fallback": used_fallback,
            },
        )

    def fallback(self, state: CrisisState, exc: Exception) -> dict[str, Any]:
        """Degrade gracefully: continue the workflow with an empty context.

        Args:
            state: Workflow state.
            exc: The exception raised.

        Returns:
            State update with an empty retrieved context.
        """
        return {"retrieved_context": RetrievedContext(used_fallback=True).model_dump(mode="json")}

    # ------------------------------------------------------------------ #
    # Retrieval steps
    # ------------------------------------------------------------------ #
    def _llm_planned_search(
        self,
        incident: IncidentData,
        analysis: AnalysisResult,
        queries: list[str],
        incidents: dict[str, RetrievedDocument],
        protocols: dict[str, RetrievedDocument],
    ) -> None:
        """Let the LLM choose search queries via tool calls, then execute them.

        Args:
            incident: Structured incident.
            analysis: Threat analysis.
            queries: Accumulator for executed queries.
            incidents: Accumulator for incident hits keyed by ID.
            protocols: Accumulator for protocol hits keyed by ID.
        """
        brief = {
            "title": incident.title,
            "category": analysis.category.value,
            "threat_type": analysis.threat_type,
            "severity": analysis.severity.value,
            "affected_assets": incident.affected_assets,
            "key_indicators": analysis.key_indicators[:8],
        }
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Incident brief:\n{self.to_json(brief)}"),
        ]
        response = self.invoke_llm(messages, self.tools, "required")
        by_name = {t.name: t for t in self.tools}
        for call in response.tool_calls[:4]:
            tool = by_name.get(call["name"])
            if tool is None:
                continue
            self._tool_trace.append(call["name"])
            try:
                hits = [RetrievedDocument.model_validate(h) for h in tool.invoke(call.get("args") or {})]
            except Exception as exc:  # noqa: BLE001 - one bad call should not sink retrieval
                logger.warning("rag: tool %s failed: %s", call["name"], exc)
                continue
            queries.append(str((call.get("args") or {}).get("query", "")))
            target = incidents if call["name"] == "search_past_incidents" else protocols
            self._merge(target, hits)

    def _fallback_search(
        self,
        incident: IncidentData,
        analysis: AnalysisResult,
        queries: list[str],
        incidents: dict[str, RetrievedDocument],
        protocols: dict[str, RetrievedDocument],
    ) -> None:
        """Run deterministic searches derived from the incident when LLM planning is insufficient.

        Args:
            incident: Structured incident.
            analysis: Threat analysis.
            queries: Accumulator for executed queries.
            incidents: Accumulator for incident hits.
            protocols: Accumulator for protocol hits.
        """
        category = analysis.category.value
        incident_query = f"{analysis.threat_type}: {incident.title}. {incident.description[:300]}"
        protocol_query = f"{analysis.threat_type} response protocol"
        if not incidents:
            queries.append(incident_query)
            self._merge(incidents, self.kb.search_incidents(incident_query, k=self.top_k_incidents, category=category))
        if not protocols:
            queries.append(protocol_query)
            self._merge(protocols, self.kb.search_protocols(protocol_query, k=self.top_k_protocols, category=category))

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    def _merge(self, target: dict[str, RetrievedDocument], hits: list[RetrievedDocument]) -> None:
        """Deduplicate hits by ID (keeping the best score) and screen their content.

        Args:
            target: Accumulator keyed by document ID.
            hits: New hits.
        """
        for hit in hits:
            if hit.similarity < self.min_similarity:
                continue
            guard = self.detector.check(hit.content)
            if guard.is_injection:
                logger.warning("rag: dropped document %s flagged for prompt injection %s", hit.doc_id, guard.matched_rules)
                continue
            hit = hit.model_copy(update={"content": hit.content[: self.MAX_DOC_CHARS]})
            existing = target.get(hit.doc_id)
            if existing is None or hit.similarity > existing.similarity:
                target[hit.doc_id] = hit

    @staticmethod
    def _rank(docs: dict[str, RetrievedDocument], limit: int) -> list[RetrievedDocument]:
        """Return the ``limit`` most similar documents.

        Args:
            docs: Documents keyed by ID.
            limit: Maximum number to keep.

        Returns:
            Documents sorted by descending similarity.
        """
        return sorted(docs.values(), key=lambda d: d.similarity, reverse=True)[:limit]

    @staticmethod
    def _confidence(context: RetrievedContext) -> float:
        """Estimate retrieval confidence from the similarity of the top documents.

        Args:
            context: Retrieved context.

        Returns:
            Confidence in ``[0, 1]``.
        """
        top_incidents = [d.similarity for d in context.past_incidents[:3]]
        top_protocols = [d.similarity for d in context.protocols[:2]]
        scores = top_incidents + top_protocols
        if not scores:
            return 0.0
        return round(max(0.0, min(1.0, sum(scores) / len(scores))), 4)
