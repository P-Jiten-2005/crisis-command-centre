"""Base agent abstraction: LLM construction, tool-calling loop and metrics."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from agents.schemas import AgentMetrics
from agents.state import CrisisState
from config import Settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class AgentError(RuntimeError):
    """Raised when an agent cannot produce a valid result."""


REASONING_MODEL_TAGS: tuple[str, ...] = ("gpt-oss", "qwen3")


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the Groq chat model used by all agents.

    Args:
        settings: Application settings.

    Returns:
        A configured ``ChatGroq`` instance.
    """
    from langchain_groq import ChatGroq

    extra: dict[str, Any] = {}
    # Reasoning models (gpt-oss, qwen3) spend most of their latency "thinking";
    # low effort is plenty for extraction/classification. Other models reject the param.
    if settings.llm_reasoning_effort and any(tag in settings.groq_model for tag in REASONING_MODEL_TAGS):
        extra["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.require_groq_key(),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
        **extra,
    )


def is_retryable_error(exc: Exception) -> bool:
    """Decide whether an LLM error is worth retrying at the agent level.

    Auth, permission, unknown-model and bad-request errors will fail identically on
    every attempt, and rate limits are already retried (with back-off) by the Groq
    SDK, so only malformed tool calls, timeouts, connection and server errors retry.

    Args:
        exc: The exception raised by the LLM call.

    Returns:
        ``True`` if another attempt may succeed.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return not isinstance(exc, (AgentError, ValueError, TypeError))
    if status == 400:
        return "tool_use_failed" in str(exc) or "tool call validation" in str(exc).lower()
    return status >= 500


def make_output_tool(name: str, description: str, schema: type[BaseModel]) -> StructuredTool:
    """Create a LangChain tool whose arguments are a Pydantic output schema.

    Forcing the model to call this tool yields structured, schema-validated JSON.

    Args:
        name: Tool name.
        description: Tool description shown to the model.
        schema: Pydantic model describing the tool arguments.

    Returns:
        The structured tool.
    """

    def _submit(**kwargs: Any) -> dict[str, Any]:
        """Validate the submitted payload against the schema."""
        return schema.model_validate(kwargs).model_dump(mode="json")

    return StructuredTool.from_function(func=_submit, name=name, description=description, args_schema=schema)


@dataclass
class TokenUsage:
    """Accumulated LLM usage for one agent execution."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def add(self, message: AIMessage) -> None:
        """Add usage reported on an AI message.

        Args:
            message: The model response.
        """
        self.llm_calls += 1
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)


@dataclass
class ToolLoopResult:
    """Result of a tool-calling loop: the validated output and the tools used."""

    output: BaseModel
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class AgentOutcome:
    """What an agent's ``run`` returns to the node wrapper."""

    update: dict[str, Any]
    confidence: float | None = None
    memory_note: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Template for all agents.

    Subclasses implement :meth:`run`. Calling the agent instance (as a LangGraph
    node) times the run, records token usage and confidence, catches errors and
    returns a partial state update.
    """

    name: ClassVar[str] = "base"

    def __init__(self, llm: BaseChatModel, max_iterations: int = 2, llm_retries: int = 2) -> None:
        """Initialise the agent.

        Args:
            llm: Chat model supporting tool calling.
            max_iterations: Attempts at a valid output-tool call (validation errors are fed back
                between attempts), on top of any helper-tool rounds.
            llm_retries: Extra attempts when an LLM call raises (e.g. malformed tool call).
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.llm_retries = llm_retries
        self._usage = TokenUsage()
        self._tool_trace: list[str] = []

    # ------------------------------------------------------------------ #
    # Node entry point
    # ------------------------------------------------------------------ #
    def __call__(self, state: CrisisState) -> dict[str, Any]:
        """Execute the agent as a LangGraph node.

        Args:
            state: Current workflow state.

        Returns:
            Partial state update including metrics and memory.
        """
        self._usage = TokenUsage()
        self._tool_trace = []
        started = time.perf_counter()
        try:
            outcome = self.run(state)
            status, error, update = "success", None, dict(outcome.update)
            confidence, note = outcome.confidence, outcome.memory_note
        except Exception as exc:  # noqa: BLE001 - node must never crash the graph
            logger.exception("Agent %s failed", self.name)
            status, error, confidence, note = "error", f"{type(exc).__name__}: {exc}", None, {}
            update = self.fallback(state, exc)
            update.setdefault("errors", []).append(f"{self.name}: {error}")

        metrics = AgentMetrics(
            agent=self.name,
            status=status,  # type: ignore[arg-type]
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            llm_calls=self._usage.llm_calls,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            total_tokens=self._usage.total_tokens,
            confidence=round(confidence, 4) if confidence is not None else None,
            tool_calls=list(self._tool_trace),
            error=error,
        )
        logger.info(
            "agent=%s status=%s latency_ms=%.0f llm_calls=%d tokens=%d confidence=%s tools=%s",
            metrics.agent,
            metrics.status,
            metrics.latency_ms,
            metrics.llm_calls,
            metrics.total_tokens,
            metrics.confidence,
            metrics.tool_calls,
        )
        update["metrics"] = [metrics.model_dump(mode="json")]
        update["memory"] = {self.name: {"status": status, "confidence": metrics.confidence, **note}}
        return update

    @abstractmethod
    def run(self, state: CrisisState) -> AgentOutcome:
        """Perform the agent's work.

        Args:
            state: Current workflow state.

        Returns:
            The agent outcome.
        """

    def fallback(self, state: CrisisState, exc: Exception) -> dict[str, Any]:
        """State update to apply when :meth:`run` raises. Override for graceful degradation.

        Args:
            state: Current workflow state.
            exc: The exception raised.

        Returns:
            A partial state update (empty by default).
        """
        return {}

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #
    def invoke_llm(self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool], tool_choice: str) -> AIMessage:
        """Invoke the LLM with bound tools, retrying transient/malformed responses.

        Args:
            messages: Conversation so far.
            tools: Tools to expose.
            tool_choice: ``"required"``, ``"auto"`` or a specific tool name.

        Returns:
            The model response.

        Raises:
            AgentError: When every attempt fails.
        """
        runnable = self.llm.bind_tools(list(tools), tool_choice=tool_choice)
        last_exc: Exception | None = None
        for attempt in range(self.llm_retries + 1):
            try:
                response = runnable.invoke(list(messages))
                if not isinstance(response, AIMessage):
                    raise AgentError(f"Unexpected LLM response type: {type(response).__name__}")
                self._usage.add(response)
                return response
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_exc = exc
                logger.warning("agent=%s LLM call failed (attempt %d): %s", self.name, attempt + 1, exc)
                if not is_retryable_error(exc):
                    raise AgentError(f"LLM call failed (not retryable): {exc}") from exc
                if attempt < self.llm_retries:
                    time.sleep(min(0.5 * 2.0**attempt, 4.0))
        raise AgentError(f"LLM call failed after {self.llm_retries + 1} attempts: {last_exc}")

    def run_tool_loop(
        self,
        messages: list[BaseMessage],
        output_tool: StructuredTool,
        output_schema: type[ModelT],
        helper_tools: Sequence[BaseTool] = (),
    ) -> ModelT:
        """Run a tool-calling loop until the model submits a valid output.

        The model may call ``helper_tools`` (whose results are fed back) for up to one
        round per helper tool before the output tool is forced. Validation errors are
        returned to the model so it can self-correct.

        Args:
            messages: Initial conversation (mutated in place).
            output_tool: Tool whose arguments are the final structured output.
            output_schema: Pydantic schema used to validate the output tool arguments.
            helper_tools: Optional informational tools.

        Returns:
            The validated output model.

        Raises:
            AgentError: If no valid output is produced within the iteration budget.
        """
        tools: list[BaseTool] = [*helper_tools, output_tool]
        by_name = {t.name: t for t in tools}
        helper_rounds = len(helper_tools)
        budget = helper_rounds + self.max_iterations
        for iteration in range(budget):
            # Some models (e.g. gpt-oss) call one helper per turn, so allow one round per
            # helper before forcing. Agents should pre-load helper info into the prompt so
            # the model can usually submit on the first call.
            force_output = iteration >= helper_rounds
            # When forcing, expose only the output tool: Groq rejects the whole request
            # (tool_use_failed) if the model attempts any other bound tool.
            if force_output:
                response = self.invoke_llm(messages, [output_tool], output_tool.name)
            else:
                response = self.invoke_llm(messages, tools, "required")
            messages.append(response)

            if not response.tool_calls:
                messages.append(HumanMessage(content=f"You must call the `{output_tool.name}` tool."))
                continue

            final_call = None
            for call in response.tool_calls:
                self._tool_trace.append(call["name"])
                if call["name"] == output_tool.name:
                    if final_call is None:
                        final_call = call
                    else:
                        messages.append(ToolMessage(content="Ignored duplicate submission.", tool_call_id=call["id"]))
                    continue
                messages.append(self._execute_tool(by_name.get(call["name"]), call))

            if final_call is None:
                continue
            try:
                return output_schema.model_validate(final_call["args"])
            except ValidationError as exc:
                logger.warning("agent=%s output validation failed: %s", self.name, exc.errors()[:3])
                messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed: {exc}. Call `{output_tool.name}` again with corrected arguments "
                            "that satisfy the schema."
                        ),
                        tool_call_id=final_call["id"],
                    )
                )
                # Any other tool calls in this turn were already answered above.
        raise AgentError(f"{self.name}: no valid `{output_tool.name}` call within {budget} iterations")

    @staticmethod
    def _execute_tool(tool: BaseTool | None, call: dict[str, Any]) -> ToolMessage:
        """Execute a helper tool call and wrap its result as a ToolMessage.

        Args:
            tool: The tool to run, or ``None`` if the model called an unknown tool.
            call: The tool call dict from the AI message.

        Returns:
            The tool result message.
        """
        if tool is None:
            content = json.dumps({"error": f"Unknown tool '{call['name']}'"})
        else:
            try:
                content = json.dumps(tool.invoke(call.get("args") or {}), default=str)
            except Exception as exc:  # noqa: BLE001 - report tool failures to the model
                content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        return ToolMessage(content=content, tool_call_id=call["id"])

    @staticmethod
    def to_json(data: Any) -> str:
        """Serialise data as compact JSON for prompts.

        Args:
            data: JSON-serialisable data.

        Returns:
            JSON string.
        """
        return json.dumps(data, ensure_ascii=False, default=str, indent=1)
