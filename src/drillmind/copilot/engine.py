"""
Copilot Engine
===============
Pluggable LLM backend for the drilling copilot.

Supports:
1. Ollama (local, default) — e.g. mistral, llama3, qwen2
2. OpenAI API — gpt-4o, gpt-4o-mini
3. Anthropic API — claude-sonnet, claude-haiku
4. Fallback — rule-based context dump (no LLM required)

The engine is responsible for:
- Building the context from application state
- Constructing the grounded prompt
- Calling the LLM
- Returning a structured response

Usage:
    engine = CopilotEngine(provider="ollama", model="mistral")
    response = await engine.query("What is the current rig state?", app_state)
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from drillmind.copilot.context_builder import build_full_context
from drillmind.copilot.prompt_templates import build_system_prompt, build_user_prompt


@dataclass
class CopilotResponse:
    """Structured response from the copilot."""

    answer: str
    provider: str
    model: str
    context_summary: dict[str, Any]
    grounded: bool = True  # True if answer is backed by data

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "grounded": self.grounded,
            "context_summary": self.context_summary,
        }


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from system + user prompts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class OllamaProvider(LLMProvider):
    """
    Local LLM via Ollama.

    Requires: ollama installed and running (https://ollama.com)
    Default model: mistral (7B, fast, good for technical Q&A)
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "No response from model.")
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider. Requires OPENAI_API_KEY env var."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._api_key = os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Anthropic API provider. Requires ANTHROPIC_API_KEY env var."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._model = model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]


class FallbackProvider(LLMProvider):
    """
    Rule-based fallback that works without any LLM.
    Produces a data-grounded response by directly interpreting
    the context using hardcoded drilling engineering rules.
    """

    @property
    def name(self) -> str:
        return "fallback"

    @property
    def model_name(self) -> str:
        return "rule-based-v1"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Parse the context from the user prompt to generate a rule-based answer
        return self._rule_based_response(user_prompt)

    def _rule_based_response(self, prompt: str) -> str:
        """Generate a response using drilling engineering rules."""
        lines = []
        question_lower = prompt.lower()

        # Extract data from the prompt
        lines.append("**DrillMind Analysis (Rule-Based Mode)**\n")

        if "anomal" in question_lower or "event" in question_lower:
            lines.append("## Anomaly Assessment")
            if "anomaly active: YES" in prompt:
                lines.append(
                    "⚠️ **An anomaly is currently active.** "
                    "Review the anomaly score and recent events below for details."
                )
            else:
                lines.append(
                    "✅ No anomaly is currently active. The drilling parameters "
                    "are within the learned normal baseline."
                )
            # Extract and echo recent events
            if "Recent Anomaly Events" in prompt:
                events_section = prompt.split("Recent Anomaly Events")[1].split("###")[0]
                lines.append(f"\n{events_section.strip()}")

        elif "state" in question_lower or "doing" in question_lower:
            lines.append("## Rig State Assessment")
            if "Current state:" in prompt:
                state_line = [l for l in prompt.split("\n") if "Current state:" in l]
                if state_line:
                    lines.append(f"The rig is currently: {state_line[0].split(':**')[1].strip()}")

        elif "mse" in question_lower or "efficiency" in question_lower:
            lines.append("## Drilling Efficiency Assessment")
            if "MSE" in prompt:
                mse_line = [l for l in prompt.split("\n") if "MSE" in l]
                for ml in mse_line:
                    lines.append(f"  {ml.strip()}")
                lines.append(
                    "\nMSE interpretation: Values 10-100 MPa are normal for North Sea formations. "
                    "Values >200 MPa suggest inefficient drilling (possible bit wear or founder point)."
                )

        elif "rop" in question_lower or "penetration" in question_lower:
            lines.append("## Rate of Penetration Assessment")
            lines.append(
                "ROP data is available during active drilling periods only. "
                "The d-exponent normalizes ROP for WOB and RPM changes."
            )

        elif "pressure" in question_lower or "kick" in question_lower:
            lines.append("## Pressure Assessment")
            if "Standpipe Pressure" in prompt:
                spp_line = [l for l in prompt.split("\n") if "Standpipe Pressure" in l]
                if spp_line:
                    lines.append(f"  {spp_line[0].strip()}")
            if "Casing Pressure" in prompt:
                cp_line = [l for l in prompt.split("\n") if "Casing Pressure" in l]
                if cp_line:
                    lines.append(f"  {cp_line[0].strip()}")
            lines.append(
                "\nMonitor pit volume and flow for kick indicators. "
                "Any sustained casing pressure increase requires immediate investigation."
            )

        else:
            # Generic: dump key context
            lines.append("## Drilling Status Summary")
            for section in ["Sensor Snapshot", "Anomaly Detection", "Rig State", "Drilling KPIs"]:
                if section in prompt:
                    section_text = prompt.split(section)[1].split("###")[0]
                    lines.append(f"\n### {section}")
                    lines.append(section_text.strip()[:500])

        lines.append(
            "\n---\n*Note: This is a rule-based response. For natural language analysis, "
            "configure an LLM provider (Ollama, OpenAI, or Anthropic).*"
        )

        return "\n".join(lines)


def create_provider(
    provider: str = "fallback",
    model: str | None = None,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Parameters
    ----------
    provider : str
        One of: "ollama", "openai", "anthropic", "fallback"
    model : str | None
        Model name. Defaults depend on provider.

    Returns
    -------
    LLMProvider
    """
    provider = provider.lower()

    if provider == "ollama":
        return OllamaProvider(model=model or "mistral")
    elif provider == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini")
    elif provider == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")
    elif provider == "fallback":
        return FallbackProvider()
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: ollama, openai, anthropic, fallback"
        )


class CopilotEngine:
    """
    Main copilot engine that orchestrates context building,
    prompt construction, and LLM generation.

    Parameters
    ----------
    provider : str
        LLM provider name.
    model : str | None
        Model name.
    """

    def __init__(
        self,
        provider: str = "fallback",
        model: str | None = None,
    ) -> None:
        self._llm = create_provider(provider, model)
        logger.info(
            "Copilot engine initialized: provider={}, model={}",
            self._llm.name,
            self._llm.model_name,
        )

    async def query(
        self,
        question: str,
        app_state: dict[str, Any],
    ) -> CopilotResponse:
        """
        Process a natural language question from an RTOC analyst.

        Parameters
        ----------
        question : str
            The analyst's question (e.g., "Why did ROP drop at 3450m?")
        app_state : dict
            Application state dict containing:
            - time_df: pd.DataFrame
            - events: list[AnomalyEvent]
            - anomaly_details: dict
            - features: pd.DataFrame
            - rig_states: pd.Series
            - transitions: pd.DataFrame
            - kpi_df: pd.DataFrame
            - settings: Settings

        Returns
        -------
        CopilotResponse
            Structured response with answer and metadata.
        """
        logger.info("Copilot query: '{}'", question[:100])

        # Build context from live data
        context = build_full_context(
            time_df=app_state["time_df"],
            events=app_state["events"],
            anomaly_details=app_state["anomaly_details"],
            features=app_state["features"],
            rig_states=app_state["rig_states"],
            transitions=app_state.get("transitions"),
            kpi_df=app_state["kpi_df"],
            settings=app_state["settings"],
        )

        # Build prompts
        system_prompt = build_system_prompt(context["well"])
        user_prompt = build_user_prompt(context, question)

        logger.debug(
            "Prompt built: system={}chars, user={}chars",
            len(system_prompt),
            len(user_prompt),
        )

        # Generate response
        try:
            answer = await self._llm.generate(system_prompt, user_prompt)
        except Exception as e:
            logger.error("LLM generation failed: {}", e)
            # Fall back to rule-based
            fallback = FallbackProvider()
            answer = await fallback.generate(system_prompt, user_prompt)
            return CopilotResponse(
                answer=answer,
                provider="fallback",
                model="rule-based-v1",
                context_summary={
                    "anomaly_score": context["anomalies"]["current_anomaly_score"],
                    "rig_state": context["rig_state"]["current_state"],
                    "total_events": context["anomalies"]["total_events"],
                },
                grounded=True,
            )

        return CopilotResponse(
            answer=answer,
            provider=self._llm.name,
            model=self._llm.model_name,
            context_summary={
                "anomaly_score": context["anomalies"]["current_anomaly_score"],
                "rig_state": context["rig_state"]["current_state"],
                "total_events": context["anomalies"]["total_events"],
            },
            grounded=True,
        )
