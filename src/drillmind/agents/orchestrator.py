"""
DrillMind — Query Orchestrator
======================================
Framework-free iterative tool-calling agent.

Architecture:
1. Router classifies user intent → selects relevant tool set
2. LLM plans tool calls based on the question
3. Tools execute and return evidence
4. LLM synthesizes final answer with evidence citations
5. Fallback: deterministic rule-based orchestration if no LLM

The orchestrator keeps the existing copilot engine as its LLM
backend — it just wraps it with tool-calling capabilities.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from drillmind.agents.tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_descriptions,
)


# ---------------------------------------------------------------------------
# Intent Router
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Classify user question intent and select relevant tools.

    Uses keyword matching for deterministic routing.
    When an LLM is available, the LLM selects tools dynamically.
    """

    # Intent → tool mappings (ordered by priority)
    INTENT_MAP = {
        "anomaly": {
            "keywords": [
                "anomal", "alert", "alarm", "unusual", "abnormal",
                "kick", "loss", "stuck", "washout", "dysfunction",
                "event", "detection", "flag", "warning",
            ],
            "tools": ["get_anomaly_status", "get_current_sensors", "search_ddr"],
        },
        "safety": {
            "keywords": [
                "kick", "well control", "shut in", "kill",
                "gas influx", "pit gain", "flow check", "BOP",
                "safety", "risk", "danger", "emergency",
            ],
            "tools": ["get_anomaly_status", "get_current_sensors", "get_rig_state", "search_ddr"],
        },
        "kpi": {
            "keywords": [
                "MSE", "specific energy", "d-exponent", "d_exp",
                "efficiency", "performance", "KPI", "bit",
                "ROP", "rate of penetration",
            ],
            "tools": ["get_drilling_kpis", "get_current_sensors", "get_rig_state", "get_rop_formation"],
        },
        "state": {
            "keywords": [
                "rig state", "what is the rig doing", "drilling",
                "circulating", "tripping", "static", "connection",
                "state", "activity", "operation",
            ],
            "tools": ["get_rig_state", "get_current_sensors"],
        },
        "historical": {
            "keywords": [
                "DDR", "report", "history", "historical", "past",
                "previous", "offset well", "compare", "when did",
                "mud weight", "formation", "casing", "cement",
                "BHA", "whipstock", "sidetrack",
            ],
            "tools": ["search_ddr", "query_production", "compare_wells", "get_depth_log"],
        },
        "quality": {
            "keywords": [
                "quality", "gap", "spike", "flatline", "data quality",
                "sensor", "missing", "sparse",
            ],
            "tools": ["get_data_quality", "get_current_sensors"],
        },
        "production": {
            "keywords": [
                "production", "oil", "gas", "water", "rate",
                "well comparison", "offset",
            ],
            "tools": ["query_production", "compare_wells"],
        },
        "formation": {
            "keywords": [
                "formation", "porosity", "permeability", "shale",
                "lithology", "gamma ray", "resistivity", "density",
                "neutron", "LWD", "MWD", "petrophysic", "log",
                "saturation", "KLOGH", "VSH",
            ],
            "tools": ["get_depth_log", "get_rop_formation", "get_drilling_kpis"],
        },
        "general": {
            "keywords": [
                "status", "summary", "overview", "current",
                "what", "how", "tell me", "report",
            ],
            "tools": ["get_current_sensors", "get_anomaly_status", "get_rig_state", "get_drilling_kpis"],
        },
    }

    @classmethod
    def classify(cls, question: str) -> tuple[str, list[str]]:
        """
        Classify question intent and return (intent, tool_list).

        Parameters
        ----------
        question : str
            User question.

        Returns
        -------
        tuple[str, list[str]]
            Intent name and list of tool names to call.
        """
        q_lower = question.lower()

        # Score each intent by keyword matches
        scores = {}
        for intent, config in cls.INTENT_MAP.items():
            score = sum(1 for kw in config["keywords"] if kw.lower() in q_lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general", cls.INTENT_MAP["general"]["tools"]

        # Return highest scoring intent
        best_intent = max(scores, key=scores.get)
        return best_intent, cls.INTENT_MAP[best_intent]["tools"]


# ---------------------------------------------------------------------------
# Evidence accumulator
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Evidence collected from tool calls."""
    tool_name: str
    result: dict
    execution_time: float

    def to_context_string(self) -> str:
        """Format evidence for LLM context."""
        # Compact JSON with key fields
        result_str = json.dumps(self.result, indent=None, default=str)
        if len(result_str) > 1500:
            result_str = result_str[:1500] + "..."
        return f"[TOOL: {self.tool_name}] {result_str}"


@dataclass
class OrchestratorResult:
    """Result from the orchestrator."""
    answer: str
    intent: str
    tools_called: list[str]
    evidence: list[Evidence]
    total_time: float
    grounded: bool = True

    def to_api_dict(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "tools_called": self.tools_called,
            "evidence_count": len(self.evidence),
            "total_time_ms": round(self.total_time * 1000),
            "grounded": self.grounded,
        }


# ---------------------------------------------------------------------------
# Agent Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Tool-calling query orchestrator for drilling data.

    Supports two modes:
    1. **LLM mode**: LLM selects tools and synthesizes answers
    2. **Rule-based mode**: Deterministic tool selection + template answers

    Parameters
    ----------
    state : dict
        Application state dict containing DataFrames and stores.
    llm_fn : callable, optional
        Async function that takes (system_prompt, user_prompt) and returns str.
        If None, falls back to rule-based mode.
    """

    def __init__(self, state: dict, llm_fn=None):
        self._state = state
        self._llm_fn = llm_fn

    async def query(self, question: str) -> OrchestratorResult:
        """
        Process a natural language question using tool-based orchestration.

        Parameters
        ----------
        question : str
            User's natural language question.

        Returns
        -------
        OrchestratorResult
            Answer with evidence chain.
        """
        t0 = time.time()

        # 1. Route intent
        intent, tools_to_call = IntentRouter.classify(question)
        logger.info(f"Agent: intent={intent}, tools={tools_to_call}")

        # 2. Execute tools and collect evidence
        evidence_list: list[Evidence] = []
        for tool_name in tools_to_call:
            t_tool = time.time()

            # Extract search query for DDR tool
            params = {}
            if tool_name == "search_ddr":
                params["query"] = question

            result = execute_tool(tool_name, self._state, **params)
            elapsed = time.time() - t_tool

            evidence_list.append(Evidence(
                tool_name=tool_name,
                result=result,
                execution_time=elapsed,
            ))

        # 3. Synthesize answer
        if self._llm_fn:
            answer = await self._synthesize_with_llm(question, intent, evidence_list)
        else:
            answer = self._synthesize_rule_based(question, intent, evidence_list)

        total_time = time.time() - t0
        logger.info(
            f"Agent complete: intent={intent}, tools={len(evidence_list)}, "
            f"time={total_time:.2f}s"
        )

        return OrchestratorResult(
            answer=answer,
            intent=intent,
            tools_called=[e.tool_name for e in evidence_list],
            evidence=evidence_list,
            total_time=total_time,
        )

    async def _synthesize_with_llm(
        self,
        question: str,
        intent: str,
        evidence: list[Evidence],
    ) -> str:
        """Synthesize answer using LLM with evidence context."""
        system_prompt = self._build_agent_system_prompt()
        user_prompt = self._build_agent_user_prompt(question, evidence)

        try:
            answer = await self._llm_fn(system_prompt, user_prompt)
            return answer
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}, falling back to rule-based")
            return self._synthesize_rule_based(question, intent, evidence)

    def _synthesize_rule_based(
        self,
        question: str,
        intent: str,
        evidence: list[Evidence],
    ) -> str:
        """Deterministic rule-based answer synthesis."""
        lines = [f"**DrillMind Agent Analysis** (intent: {intent})\n"]

        for ev in evidence:
            result = ev.result

            if ev.tool_name == "get_current_sensors":
                sensors = result.get("sensors", {})
                if sensors:
                    lines.append("## Current Sensor Readings")
                    for key, val in sensors.items():
                        if key != "timestamp" and val is not None:
                            lines.append(f"- **{key}**: {val}")
                    ts = sensors.get("timestamp")
                    if ts:
                        lines.append(f"- *Timestamp*: {ts}")

            elif ev.tool_name == "get_anomaly_status":
                score = result.get("score", 0)
                active = result.get("anomaly_active", False)
                lines.append("## Anomaly Status")
                if active:
                    lines.append(f"⚠️ **Anomaly ACTIVE** — score: {score}")
                else:
                    lines.append(f"✅ No active anomaly — score: {score}")

                events = result.get("recent_events", [])
                if events:
                    lines.append(f"\n**Top {len(events)} events:**")
                    for evt in events:
                        lines.append(
                            f"- [{evt['severity'].upper()}] {evt['type']} at {evt['timestamp']} "
                            f"(score: {evt['score']}) — {evt['description'][:100]}"
                        )

            elif ev.tool_name == "get_rig_state":
                state = result.get("current_state", "unknown")
                lines.append(f"## Rig State: **{state.upper()}**")
                breakdown = result.get("breakdown", {})
                if breakdown:
                    for s, info in sorted(breakdown.items(), key=lambda x: -x[1]["pct"]):
                        lines.append(f"- {s}: {info['pct']}% ({info['count']} samples)")

            elif ev.tool_name == "get_drilling_kpis":
                kpis = result.get("kpis", {})
                lines.append("## Drilling KPIs")
                for kpi_name, kpi_val in kpis.items():
                    if kpi_val and isinstance(kpi_val, dict):
                        lines.append(
                            f"- **{kpi_name}**: current={kpi_val.get('current')}, "
                            f"mean={kpi_val.get('mean')}, range=[{kpi_val.get('min')}, {kpi_val.get('max')}], "
                            f"valid samples={kpi_val.get('valid_count')}"
                        )
                    elif kpi_val is None:
                        lines.append(f"- **{kpi_name}**: Not available (no active drilling)")
                note = result.get("note")
                if note:
                    lines.append(f"*Note: {note}*")

            elif ev.tool_name == "search_ddr":
                ddr_results = result.get("results", [])
                if ddr_results:
                    lines.append(f"## DDR Search Results ({len(ddr_results)} matches)")
                    for r in ddr_results:
                        lines.append(f"- **{r['source']}**")
                        text = r["text"][:200]
                        lines.append(f"  > {text}...")
                elif result.get("error"):
                    lines.append(f"## DDR Search: {result['error']}")

            elif ev.tool_name == "get_data_quality":
                lines.append("## Data Quality")
                lines.append(f"- Rows: {result.get('total_rows', '?')}")
                lines.append(f"- Time gaps: {result.get('time_gaps', '?')}")
                lines.append(f"- Spikes: {result.get('spikes_detected', '?')}")
                lines.append(f"- Flatlines: {result.get('flatline_segments', '?')}")

            elif ev.tool_name == "query_production":
                wells = result.get("wells", [])
                lines.append(f"## Production Data ({len(wells)} wells)")
                for w in wells[:7]:
                    lines.append(f"- {w}")

            elif ev.tool_name == "compare_wells":
                comparison = result.get("comparison", {})
                if comparison:
                    lines.append(f"## Well Comparison ({len(comparison)} wells)")
                    for well, data in comparison.items():
                        metrics = data.get("metrics", {})
                        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                        lines.append(f"- **{well}**: {metric_str}")

        lines.append("\n---")
        lines.append("*Mode: rule-based | Configure Ollama/OpenAI for LLM-backed answers*")

        return "\n".join(lines)

    def _build_agent_system_prompt(self) -> str:
        """Build system prompt for LLM agent."""
        tools_desc = get_tool_descriptions()
        return f"""You are DrillMind, a drilling data analysis system for Real-Time Operations Centers (RTOC) in oil & gas drilling.

You have access to the following tools that have already been called to gather evidence:

{tools_desc}

RULES:
1. Base EVERY statement on the tool evidence provided. Never fabricate data.
2. Cite your sources: "[from get_anomaly_status]", "[from search_ddr: DDR #37]"
3. Use drilling domain terminology correctly (IADC rig states, SPE conventions).
4. If the evidence is insufficient, say so — do not speculate.
5. Recommend specific actions when anomalies or risks are identified.
6. Format your response with clear headers and bullet points.

WELL CONTEXT: Equinor Volve Field, North Sea, Well 15/9-F-9 A, 12¼" section."""

    def _build_agent_user_prompt(self, question: str, evidence: list[Evidence]) -> str:
        """Build user prompt with evidence context."""
        evidence_text = "\n\n".join(e.to_context_string() for e in evidence)
        return f"""QUESTION: {question}

COLLECTED EVIDENCE:
{evidence_text}

Based on the evidence above, provide a detailed, evidence-based answer to the question.
Cite tool names and DDR references where applicable."""
