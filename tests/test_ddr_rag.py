"""
DrillMind — Tests for DDR Parser, RAG Pipeline, and Agent Orchestrator.
"""

import sys
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# DDR Parser Tests
# ---------------------------------------------------------------------------

class TestDDRParser:
    """Test DDR document parsing from HuggingFace dataset."""

    @pytest.fixture(scope="class")
    def ddrs(self):
        from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
        return load_ddrs_from_huggingface(max_docs=20)

    def test_loads_documents(self, ddrs):
        assert len(ddrs) > 0, "Should load at least some DDR documents"

    def test_doc_has_required_fields(self, ddrs):
        doc = ddrs[0]
        assert doc.doc_id, "doc_id should not be empty"
        assert doc.well_name, "well_name should not be empty"
        assert doc.activities_text, "activities_text should not be empty"
        assert isinstance(doc.report_index, int)

    def test_activities_parsed(self, ddrs):
        docs_with_activities = [d for d in ddrs if len(d.activities) > 0]
        assert len(docs_with_activities) > 0, "Some DDRs should have parsed activities"

    def test_depths_extracted(self, ddrs):
        docs_with_depth = [d for d in ddrs if len(d.depths_mentioned) > 0]
        assert len(docs_with_depth) > 0, "Some DDRs should mention depths"

    def test_operations_classified(self, ddrs):
        all_ops = set(op for d in ddrs for op in d.operations)
        assert len(all_ops) >= 3, f"Should find multiple operation types, got: {all_ops}"

    def test_metadata_dict(self, ddrs):
        meta = ddrs[0].metadata
        assert "well_name" in meta
        assert "report_index" in meta
        assert "char_count" in meta


# ---------------------------------------------------------------------------
# Chunker Tests
# ---------------------------------------------------------------------------

class TestChunker:
    """Test drilling-specific DDR chunking."""

    @pytest.fixture(scope="class")
    def chunks(self):
        from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
        from drillmind.rag.chunker import chunk_all_ddrs

        ddrs = load_ddrs_from_huggingface(max_docs=10)
        return chunk_all_ddrs(ddrs)

    def test_produces_chunks(self, chunks):
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self, chunks):
        c = chunks[0]
        assert c.chunk_id
        assert c.doc_id
        assert c.text
        assert c.chunk_type in ("activity", "summary")

    def test_has_both_chunk_types(self, chunks):
        types = set(c.chunk_type for c in chunks)
        assert "activity" in types or "summary" in types

    def test_metadata_valid(self, chunks):
        meta = chunks[0].metadata
        assert "doc_id" in meta
        assert "well_name" in meta
        assert "chunk_type" in meta


# ---------------------------------------------------------------------------
# RAG Store Tests
# ---------------------------------------------------------------------------

class TestRAGStore:
    """Test ChromaDB vector store for DDR embeddings."""

    PERSIST_DIR = "data/chromadb_pytest"

    @pytest.fixture(scope="class")
    def store(self):
        from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
        from drillmind.rag.chunker import chunk_all_ddrs
        from drillmind.rag.store import DDRVectorStore

        ddrs = load_ddrs_from_huggingface(max_docs=10)
        chunks = chunk_all_ddrs(ddrs)

        store = DDRVectorStore(persist_dir=self.PERSIST_DIR)
        store.index_chunks(chunks)
        yield store

        # Cleanup
        shutil.rmtree(self.PERSIST_DIR, ignore_errors=True)

    def test_count_positive(self, store):
        assert store.count > 0

    def test_search_returns_results(self, store):
        results = store.search("drilling operations", top_k=3)
        assert len(results) > 0

    def test_result_has_attribution(self, store):
        results = store.search("casing cement")
        if results:
            r = results[0]
            assert r.source, "Result should have source attribution"
            assert r.well_name, "Result should have well name"

    def test_search_relevance(self, store):
        results = store.search("whipstock sidetrack window")
        if results:
            # At least one result should contain whipstock-related text
            texts = " ".join(r.text.lower() for r in results)
            assert any(kw in texts for kw in ["whipstock", "window", "sidetrack", "milling"]), \
                "Search results should be relevant to query"


# ---------------------------------------------------------------------------
# Agent Tools Tests
# ---------------------------------------------------------------------------

class TestAgentTools:
    """Test domain tools."""

    def test_tool_registry_populated(self):
        from drillmind.agents.tools import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) >= 8, f"Should have at least 8 tools, got {len(TOOL_REGISTRY)}"

    def test_get_tool_descriptions(self):
        from drillmind.agents.tools import get_tool_descriptions
        desc = get_tool_descriptions()
        assert "get_current_sensors" in desc
        assert "search_ddr" in desc

    def test_execute_unknown_tool(self):
        from drillmind.agents.tools import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert "error" in result


# ---------------------------------------------------------------------------
# Agent Orchestrator Tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    """Test query orchestrator."""

    def test_intent_classification(self):
        from drillmind.agents.orchestrator import IntentRouter

        intent, tools = IntentRouter.classify("Are there any anomalies right now?")
        assert intent == "anomaly"
        assert "get_anomaly_status" in tools

    def test_kpi_intent(self):
        from drillmind.agents.orchestrator import IntentRouter

        intent, _ = IntentRouter.classify("Show me the MSE and d-exponent KPI values")
        assert intent == "kpi"

    def test_historical_intent(self):
        from drillmind.agents.orchestrator import IntentRouter

        intent, tools = IntentRouter.classify("What mud weight was used in the DDR reports?")
        assert intent == "historical"
        assert "search_ddr" in tools

    def test_safety_intent(self):
        from drillmind.agents.orchestrator import IntentRouter

        intent, _ = IntentRouter.classify("Is there a kick risk? Check well control status")
        assert intent == "safety"

    def test_general_fallback(self):
        from drillmind.agents.orchestrator import IntentRouter

        intent, _ = IntentRouter.classify("hello")
        assert intent == "general"
