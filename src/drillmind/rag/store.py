"""
DrillMind — RAG Vector Store (ChromaDB)
========================================
Persistent vector store for DDR chunk embeddings.

Architecture:
- ChromaDB for persistent storage with metadata filtering
- sentence-transformers for embedding (all-MiniLM-L6-v2)
- Hybrid search: metadata filter + semantic similarity
- Evidence attribution: every result carries source reference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from drillmind.rag.chunker import DDRChunk


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """A single search result with evidence attribution."""
    chunk_id: str
    text: str
    score: float               # Similarity score (lower = more similar for L2)
    well_name: str
    report_index: int
    chunk_type: str
    operations: str
    source: str                # Human-readable citation

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "well_name": self.well_name,
            "report_index": self.report_index,
            "chunk_type": self.chunk_type,
            "operations": self.operations,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class DDRVectorStore:
    """
    ChromaDB-backed vector store for DDR embeddings.

    Parameters
    ----------
    persist_dir : Path
        Directory for ChromaDB persistence.
    collection_name : str
        ChromaDB collection name.
    embedding_model : str
        sentence-transformers model name.
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chromadb",
        collection_name: str = "volve_ddrs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embed_fn = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB client and embedding function."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB is required for RAG. Install with: "
                "pip install chromadb sentence-transformers"
            )

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )

        # Use sentence-transformers embedding function
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model_name
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB initialized: {self._persist_dir}, "
            f"collection='{self._collection_name}', "
            f"existing docs={self._collection.count()}"
        )

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        self._ensure_initialized()
        return self._collection.count()

    def index_chunks(
        self,
        chunks: list[DDRChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Index DDR chunks into ChromaDB.

        Parameters
        ----------
        chunks : list[DDRChunk]
            Chunks to embed and store.
        batch_size : int
            Batch size for upsert operations.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        self._ensure_initialized()

        # Check for existing docs to avoid re-indexing
        existing = self._collection.count()
        if existing > 0:
            logger.info(
                f"Collection already has {existing} docs. "
                f"Upserting {len(chunks)} chunks (dedup by ID)."
            )

        t0 = time.time()
        indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            indexed += len(batch)

            if indexed % 500 == 0 or indexed == len(chunks):
                logger.debug(f"Indexed {indexed}/{len(chunks)} chunks...")

        elapsed = time.time() - t0
        logger.info(
            f"Indexing complete: {indexed} chunks in {elapsed:.1f}s "
            f"({indexed / max(elapsed, 0.1):.0f} chunks/sec)"
        )

        return indexed

    def search(
        self,
        query: str,
        top_k: int = 5,
        well_filter: Optional[str] = None,
        operation_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[RAGResult]:
        """
        Semantic search over DDR chunks.

        Parameters
        ----------
        query : str
            Natural language search query.
        top_k : int
            Number of results to return.
        well_filter : str, optional
            Filter by well name (exact match).
        operation_filter : str, optional
            Filter by operation type (substring match).
        chunk_type : str, optional
            Filter by chunk type ("activity" or "summary").

        Returns
        -------
        list[RAGResult]
            Search results with evidence attribution.
        """
        self._ensure_initialized()

        if self._collection.count() == 0:
            logger.warning("RAG store is empty — no DDRs indexed")
            return []

        # Build metadata filter
        where_filter = None
        conditions = []

        if well_filter:
            conditions.append({"well_name": {"$eq": well_filter}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})
        if operation_filter:
            conditions.append({"operations": {"$contains": operation_filter}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # Parse results
        rag_results: list[RAGResult] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return rag_results

        for idx, (doc_id, text, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            well = meta.get("well_name", "Unknown")
            rpt_idx = meta.get("report_index", -1)
            ctype = meta.get("chunk_type", "activity")
            ops = meta.get("operations", "")

            # Human-readable citation
            source = f"DDR #{rpt_idx} — Well: {well} — Type: {ctype}"
            if ops:
                source += f" — Ops: {ops}"

            rag_results.append(RAGResult(
                chunk_id=doc_id,
                text=text,
                score=dist,
                well_name=well,
                report_index=rpt_idx,
                chunk_type=ctype,
                operations=ops,
                source=source,
            ))

        return rag_results

    def delete_collection(self) -> None:
        """Delete the entire collection (for re-indexing)."""
        self._ensure_initialized()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self._collection_name}' deleted and recreated")
