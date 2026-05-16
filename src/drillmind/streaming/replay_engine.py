"""
Real-Time Replay Engine
========================
WebSocket server that replays the parsed time-indexed drilling DataFrame
as a simulated real-time stream — exactly as a WITSML server would feed
data to an RTOC dashboard.

Features
--------
- Configurable speed multiplier (1x = real-time, 10x = accelerated)
- Broadcasts to all connected WebSocket clients simultaneously
- Emits JSON messages with standardized column names
- Supports control messages: pause, resume, set_speed, seek
- Graceful handling of client connect/disconnect

Architecture
------------
1. Load the parsed DataFrame into memory (from ``load_time_log``)
2. Iterate through rows, computing the real time delta between consecutive rows
3. Sleep for (delta / speed_multiplier) between emissions
4. Broadcast each row as a JSON object to all connected clients
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import websockets
from loguru import logger

from drillmind.config import get_settings


class ReplayEngine:
    """
    WebSocket server that replays drilling telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame from ``load_time_log()``.
    host : str
        WebSocket bind host.
    port : int
        WebSocket bind port.
    speed : int
        Initial speed multiplier.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        host: str | None = None,
        port: int | None = None,
        speed: int | None = None,
    ) -> None:
        settings = get_settings().replay
        self.df = df
        self.host = host or settings.websocket_host
        self.port = port or settings.websocket_port
        self.speed = speed or settings.speed_multiplier

        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self._paused = False
        self._current_index = 0
        self._running = False

    async def _register(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        logger.info("Client connected ({} total)", len(self.clients))
        # Send initial metadata
        await ws.send(json.dumps({
            "type": "meta",
            "well": get_settings().well,
            "field": get_settings().field_name,
            "total_rows": len(self.df),
            "time_start": str(self.df.index.min()),
            "time_end": str(self.df.index.max()),
            "columns": list(self.df.columns),
            "speed": self.speed,
        }))

    async def _unregister(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.discard(ws)
        logger.info("Client disconnected ({} remaining)", len(self.clients))

    async def _broadcast(self, message: str) -> None:
        if not self.clients:
            return
        disconnected = set()
        for ws in self.clients:
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                disconnected.add(ws)
        self.clients -= disconnected

    def _row_to_json(self, idx: int) -> str:
        """Serialize a single row to JSON, handling NaN and numpy types."""
        row = self.df.iloc[idx]
        timestamp = self.df.index[idx]

        data: dict[str, Any] = {
            "type": "data",
            "index": idx,
            "timestamp": str(timestamp),
        }

        for col in self.df.columns:
            val = row[col]
            if pd.isna(val):
                data[col] = None
            elif isinstance(val, (np.integer,)):
                data[col] = int(val)
            elif isinstance(val, (np.floating,)):
                data[col] = round(float(val), 6)
            elif isinstance(val, (np.bool_,)):
                data[col] = bool(val)
            else:
                data[col] = val

        return json.dumps(data)

    async def _handle_control(self, ws: websockets.WebSocketServerProtocol) -> None:
        """Listen for control messages from a client."""
        try:
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue

                action = cmd.get("action")
                if action == "pause":
                    self._paused = True
                    logger.info("Replay paused")
                    await self._broadcast(json.dumps({"type": "control", "status": "paused"}))

                elif action == "resume":
                    self._paused = False
                    logger.info("Replay resumed")
                    await self._broadcast(json.dumps({"type": "control", "status": "running"}))

                elif action == "set_speed":
                    new_speed = cmd.get("speed", self.speed)
                    if isinstance(new_speed, (int, float)) and new_speed > 0:
                        self.speed = int(new_speed)
                        logger.info("Speed set to {}x", self.speed)
                        await self._broadcast(json.dumps({
                            "type": "control",
                            "status": "speed_changed",
                            "speed": self.speed,
                        }))

                elif action == "seek":
                    target = cmd.get("index", 0)
                    if 0 <= target < len(self.df):
                        self._current_index = target
                        logger.info("Seeked to index {}", target)

        except websockets.ConnectionClosed:
            pass
        finally:
            await self._unregister(ws)

    async def _handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        """Handle a single WebSocket connection."""
        await self._register(ws)
        await self._handle_control(ws)

    async def _stream_loop(self) -> None:
        """Main loop that iterates through the DataFrame and broadcasts rows."""
        logger.info(
            "Starting replay: {} rows at {}x speed",
            len(self.df),
            self.speed,
        )
        self._running = True
        self._current_index = 0

        while self._current_index < len(self.df) and self._running:
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            message = self._row_to_json(self._current_index)
            await self._broadcast(message)

            # Compute sleep based on real time delta
            if self._current_index + 1 < len(self.df):
                t_current = self.df.index[self._current_index]
                t_next = self.df.index[self._current_index + 1]
                delta_seconds = (t_next - t_current).total_seconds()

                # Clamp delta to avoid huge sleeps from data gaps
                delta_seconds = max(0, min(delta_seconds, 60))
                sleep_time = delta_seconds / self.speed

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self._current_index += 1

        logger.info("Replay completed — {} rows streamed", self._current_index)
        await self._broadcast(json.dumps({"type": "control", "status": "completed"}))

    async def start(self) -> None:
        """Start the WebSocket server and begin streaming."""
        logger.info("Replay engine starting on ws://{}:{}", self.host, self.port)

        async with websockets.serve(self._handler, self.host, self.port):
            # Wait briefly for clients to connect before starting the stream
            logger.info("Waiting for clients... (streaming starts in 3s)")
            await asyncio.sleep(3)
            await self._stream_loop()

    def stop(self) -> None:
        """Signal the streaming loop to stop."""
        self._running = False
        logger.info("Replay engine stopped")
