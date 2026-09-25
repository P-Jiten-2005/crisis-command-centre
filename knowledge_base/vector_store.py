"""ChromaDB-backed knowledge base of past incidents and response protocols.

Embeddings are computed locally with sentence-transformers and the collections
are persisted to disk at ``CHROMA_PERSIST_DIR``.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agents.schemas import RetrievedDocument
from config import get_settings

logger = logging.getLogger(__name__)

MetadataValue = str | int | float | bool


class KnowledgeBase:
    """Semantic search over two persistent Chroma collections."""

    INCIDENTS_COLLECTION = "past_incidents"
    PROTOCOLS_COLLECTION = "response_protocols"

    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        """Open (or create) the persistent Chroma collections.

        Args:
            persist_dir: Directory where Chroma stores its data.
            embedding_model: sentence-transformers model name used for embeddings.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._embedding_fn = self._load_embedding_function(embedding_model)
        self.incidents = self._get_collection(self.INCIDENTS_COLLECTION)
        self.protocols = self._get_collection(self.PROTOCOLS_COLLECTION)
        logger.info("Knowledge base opened at %s (%s)", persist_dir, self.counts())

    @staticmethod
    def _load_embedding_function(model_name: str) -> SentenceTransformerEmbeddingFunction:
        """Load the embedding model, preferring the local cache.

        Loading with ``local_files_only`` skips several Hugging Face Hub network
        round-trips (~4-5 s per startup); the Hub is only contacted on first download.

        Args:
            model_name: sentence-transformers model name.

        Returns:
            The Chroma embedding function.
        """
        try:
            return SentenceTransformerEmbeddingFunction(model_name=model_name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            logger.info("Embedding model %s not cached locally; downloading", model_name)
            return SentenceTransformerEmbeddingFunction(model_name=model_name)

    def _get_collection(self, name: str) -> Any:
        """Get or create a cosine-distance collection.

        Args:
            name: Collection name.

        Returns:
            The Chroma collection.
        """
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        doc_type: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Embed and upsert documents into the collection for ``doc_type``.

        Args:
            doc_type: ``"incident"`` or ``"protocol"``.
            ids: Unique document IDs.
            documents: Text to embed.
            metadatas: Metadata dicts (non-scalar values are serialised).

        Returns:
            Number of documents upserted.
        """
        collection = self._collection_for(doc_type)
        clean_meta = [self._sanitize_metadata(m) for m in metadatas]
        with self._lock:
            collection.upsert(ids=ids, documents=documents, metadatas=clean_meta)
        return len(ids)

    def reset(self) -> None:
        """Delete and recreate both collections."""
        with self._lock:
            for name in (self.INCIDENTS_COLLECTION, self.PROTOCOLS_COLLECTION):
                try:
                    self._client.delete_collection(name)
                except Exception:  # noqa: BLE001 - collection may not exist
                    pass
            self.incidents = self._get_collection(self.INCIDENTS_COLLECTION)
            self.protocols = self._get_collection(self.PROTOCOLS_COLLECTION)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def search_incidents(self, query: str, k: int = 4, category: str | None = None) -> list[RetrievedDocument]:
        """Semantic search over past incidents.

        Args:
            query: Natural-language search query.
            k: Maximum number of results.
            category: Optional category filter; falls back to unfiltered if it yields nothing.

        Returns:
            Matching documents ordered by similarity.
        """
        where = {"category": category} if category else None
        hits = self._query(self.incidents, "incident", query, k, where)
        if not hits and where:
            hits = self._query(self.incidents, "incident", query, k, None)
        return hits

    def search_protocols(self, query: str, k: int = 3, category: str | None = None) -> list[RetrievedDocument]:
        """Semantic search over response protocols.

        Args:
            query: Natural-language search query.
            k: Maximum number of results.
            category: Optional category filter; falls back to unfiltered if it yields nothing.

        Returns:
            Matching documents ordered by similarity.
        """
        where = {"category": category} if category else None
        hits = self._query(self.protocols, "protocol", query, k, where)
        if not hits and where:
            hits = self._query(self.protocols, "protocol", query, k, None)
        return hits

    def counts(self) -> dict[str, int]:
        """Return the number of documents per collection."""
        return {"incidents": self.incidents.count(), "protocols": self.protocols.count()}

    def is_empty(self) -> bool:
        """Return ``True`` when either collection has no documents."""
        counts = self.counts()
        return counts["incidents"] == 0 or counts["protocols"] == 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _collection_for(self, doc_type: str) -> Any:
        """Map a document type to its collection.

        Args:
            doc_type: ``"incident"`` or ``"protocol"``.

        Returns:
            The Chroma collection.

        Raises:
            ValueError: For unknown document types.
        """
        if doc_type == "incident":
            return self.incidents
        if doc_type == "protocol":
            return self.protocols
        raise ValueError(f"Unknown doc_type: {doc_type}")

    def _query(
        self,
        collection: Any,
        doc_type: str,
        query: str,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedDocument]:
        """Run a similarity query and convert results to :class:`RetrievedDocument`.

        Args:
            collection: Chroma collection to query.
            doc_type: Document type label for results.
            query: Query text.
            k: Maximum number of results.
            where: Optional metadata filter.

        Returns:
            Retrieved documents.
        """
        total = collection.count()
        if total == 0 or not query.strip():
            return []
        result = collection.query(
            query_texts=[query],
            n_results=max(1, min(k, total)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs: list[RetrievedDocument] = []
        for doc_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            meta = dict(meta or {})
            docs.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    doc_type=doc_type,  # type: ignore[arg-type]
                    title=str(meta.get("title", doc_id)),
                    content=text or "",
                    similarity=round(1.0 - float(dist), 4),
                    metadata=meta,
                )
            )
        return docs

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
        """Coerce metadata into Chroma-compatible scalar values.

        Args:
            metadata: Arbitrary metadata.

        Returns:
            Metadata with ``None`` dropped and lists joined into strings.
        """
        clean: dict[str, MetadataValue] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                clean[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Return the process-wide knowledge base built from settings.

    Returns:
        The singleton :class:`KnowledgeBase`.
    """
    settings = get_settings()
    return KnowledgeBase(settings.chroma_persist_dir, settings.embedding_model)
