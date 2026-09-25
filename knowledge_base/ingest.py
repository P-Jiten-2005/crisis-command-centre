"""Embed and ingest sample incidents and response protocols into ChromaDB.

Usage::

    python -m knowledge_base.ingest            # upsert (idempotent)
    python -m knowledge_base.ingest --reset    # wipe collections first
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.schemas import IncidentCategory, Severity

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


class PastResponse(BaseModel):
    """How a historical incident was handled."""

    actions: list[str] = Field(..., min_length=1)
    outcome: str
    response_time_hours: float | None = None
    lessons_learned: list[str] = Field(default_factory=list)


class IncidentRecord(BaseModel):
    """Schema of one entry in ``incidents.json``."""

    id: str
    title: str
    category: IncidentCategory
    threat_type: str
    severity: Severity
    date: str
    location: str
    description: str
    affected_assets: list[str] = Field(default_factory=list)
    people_affected: int | None = None
    past_response: PastResponse
    tags: list[str] = Field(default_factory=list)

    def to_document(self) -> str:
        """Render the record as the text that gets embedded.

        Returns:
            Document text (most discriminative content first).
        """
        response = self.past_response
        return "\n".join(
            [
                f"{self.title}",
                f"Category: {self.category.value} | Threat type: {self.threat_type} | Severity: {self.severity.value}",
                f"Location: {self.location} | Date: {self.date}",
                f"Description: {self.description}",
                "Past response actions: " + "; ".join(response.actions),
                f"Outcome: {response.outcome}",
                "Lessons learned: " + "; ".join(response.lessons_learned),
            ]
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return the metadata stored alongside the embedding."""
        return {
            "title": self.title,
            "category": self.category.value,
            "threat_type": self.threat_type,
            "severity": self.severity.value,
            "date": self.date,
            "location": self.location,
            "people_affected": self.people_affected,
            "response_time_hours": self.past_response.response_time_hours,
            "tags": self.tags,
        }


class ProtocolRecord(BaseModel):
    """Schema of one entry in ``protocols.json``."""

    id: str
    title: str
    category: IncidentCategory
    applies_to: list[str] = Field(..., min_length=1)
    steps: list[str] = Field(..., min_length=1)
    key_contacts: list[str] = Field(default_factory=list)
    notes: str = ""

    def to_document(self) -> str:
        """Render the protocol as the text that gets embedded.

        Returns:
            Document text.
        """
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(self.steps, start=1))
        return (
            f"{self.title}\nApplies to: {', '.join(self.applies_to)}\nCategory: {self.category.value}\n"
            f"Steps:\n{steps}\nKey contacts: {', '.join(self.key_contacts)}\nNotes: {self.notes}"
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return the metadata stored alongside the embedding."""
        return {"title": self.title, "category": self.category.value, "applies_to": self.applies_to}


class KnowledgeBaseIngestor:
    """Loads, validates and ingests JSON seed data into the knowledge base."""

    def __init__(self, knowledge_base: Any, data_dir: Path = DATA_DIR) -> None:
        """Create an ingestor.

        Args:
            knowledge_base: A :class:`knowledge_base.vector_store.KnowledgeBase`.
            data_dir: Directory containing ``incidents.json`` and ``protocols.json``.
        """
        self.kb = knowledge_base
        self.data_dir = data_dir

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        """Load a JSON array from disk.

        Args:
            path: File path.

        Returns:
            The parsed list.

        Raises:
            ValueError: If the file does not contain a JSON array.
        """
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        return data

    def load_incidents(self) -> list[IncidentRecord]:
        """Load and validate ``incidents.json``."""
        return [IncidentRecord.model_validate(r) for r in self._load_json(self.data_dir / "incidents.json")]

    def load_protocols(self) -> list[ProtocolRecord]:
        """Load and validate ``protocols.json`` (optional file)."""
        path = self.data_dir / "protocols.json"
        if not path.exists():
            return []
        return [ProtocolRecord.model_validate(r) for r in self._load_json(path)]

    def ingest(self, reset: bool = False) -> dict[str, int]:
        """Validate and upsert all seed data.

        Args:
            reset: Whether to wipe the collections first.

        Returns:
            Number of documents ingested per type.
        """
        incidents = self.load_incidents()
        protocols = self.load_protocols()
        if reset:
            self.kb.reset()
        n_incidents = self.kb.upsert(
            "incident",
            ids=[r.id for r in incidents],
            documents=[r.to_document() for r in incidents],
            metadatas=[r.to_metadata() for r in incidents],
        )
        n_protocols = (
            self.kb.upsert(
                "protocol",
                ids=[p.id for p in protocols],
                documents=[p.to_document() for p in protocols],
                metadatas=[p.to_metadata() for p in protocols],
            )
            if protocols
            else 0
        )
        logger.info("Ingested %d incidents and %d protocols", n_incidents, n_protocols)
        return {"incidents": n_incidents, "protocols": n_protocols}


def ingest_if_empty(knowledge_base: Any) -> dict[str, int] | None:
    """Ingest the seed data only when the knowledge base is empty.

    Args:
        knowledge_base: The knowledge base to populate.

    Returns:
        Ingestion counts, or ``None`` if nothing was ingested.
    """
    if not knowledge_base.is_empty():
        return None
    return KnowledgeBaseIngestor(knowledge_base).ingest()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Ingest crisis knowledge into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Delete existing collections before ingesting.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with the JSON seed files.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from knowledge_base.vector_store import get_knowledge_base

    try:
        kb = get_knowledge_base()
        counts = KnowledgeBaseIngestor(kb, args.data_dir).ingest(reset=args.reset)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    print(f"Ingestion complete: {counts} -> collection totals {kb.counts()} at {kb.persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
