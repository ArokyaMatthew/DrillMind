"""End-to-end RAG pipeline test script."""
import sys
sys.path.insert(0, "src")

from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
from drillmind.rag.chunker import chunk_all_ddrs
from drillmind.rag.store import DDRVectorStore

# 1. Load DDRs (small batch for test)
docs = load_ddrs_from_huggingface(max_docs=50)
print(f"\nLoaded {len(docs)} DDRs")
print(f"  With depths: {sum(len(d.depths_mentioned) > 0 for d in docs)}")
print(f"  With mud weights: {sum(len(d.mud_weights) > 0 for d in docs)}")
ops = set(op for d in docs for op in d.operations)
print(f"  Operations found: {ops}")

# 2. Chunk
chunks = chunk_all_ddrs(docs)
print(f"\nChunked into {len(chunks)} chunks")

# 3. Index
store = DDRVectorStore(persist_dir="data/chromadb_test")
store.index_chunks(chunks)
print(f"Indexed: {store.count} docs in ChromaDB")

# 4. Search tests
queries = [
    "stuck pipe torque spike",
    "mud weight drilling fluid",
    "casing cement shoe track",
    "BOP test pressure",
    "whipstock sidetrack window",
]

for q in queries:
    print(f"\n=== Search: {q!r} ===")
    results = store.search(q, top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] {r.source}")
        print(f"    {r.text[:120]}...")

# Cleanup
import shutil
shutil.rmtree("data/chromadb_test", ignore_errors=True)
print("\nDone - test DB cleaned up")
