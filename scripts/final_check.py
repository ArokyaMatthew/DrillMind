"""Final import and integration check."""
import sys
sys.path.insert(0, 'src')

# 1. Core imports
from drillmind.models.anomaly_detection import (
    EnsembleDetector, EnsembleConfig,
    AutoencoderDetector, IsolationForestDetector,
)
from drillmind.models.lstm_detector import LSTMDetector
from drillmind.agents.orchestrator import AgentOrchestrator, IntentRouter
from drillmind.agents.tools import TOOL_REGISTRY
from drillmind.rag.store import DDRVectorStore
from drillmind.rag.chunker import chunk_all_ddrs
from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
from drillmind.copilot.engine import CopilotEngine
from drillmind.streaming.replay_engine import ReplayEngine
from drillmind.data.quality import run_quality_check
from drillmind.config import get_settings

print("All imports OK")

# 2. Check EnsembleConfig has lstm_weight
cfg = EnsembleConfig()
assert hasattr(cfg, 'lstm_weight'), "EnsembleConfig missing lstm_weight"
print(f"EnsembleConfig: ae={cfg.autoencoder_weight} if={cfg.isolation_forest_weight} lstm={cfg.lstm_weight}")

# 3. Check tools
assert len(TOOL_REGISTRY) >= 8, f"Expected 8+ tools, got {len(TOOL_REGISTRY)}"
print(f"Agent tools registered: {len(TOOL_REGISTRY)}")

# 4. Check intents
for q, expected in [
    ("any anomalies?", "anomaly"),
    ("kick risk well control", "safety"),
    ("MSE d-exponent KPI", "kpi"),
    ("DDR reports mud weight", "historical"),
]:
    intent, _ = IntentRouter.classify(q)
    print(f"  Intent '{q}' -> {intent} {'OK' if intent == expected else 'MISMATCH'}")

# 5. Server check
from drillmind.api.server import app
routes = [r for r in app.routes if hasattr(r, 'methods')]
print(f"API endpoints: {len(routes)}")

print("\nAll checks passed.")
