"""
FastAPI + WebSocket dashboard for The Sanctuary simulation.

Supports two modes:
  - Live mode (Mode 2): watches a running simulation, receives broadcasts
  - Replay mode (Mode 3): loads a completed run directory, serves timeline

The engine sets _dashboard_broadcast via set_engine() to push state
updates to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from sanctuary.protocols.factory import list_protocols

app = FastAPI(title="The Sanctuary")

_engine: Any = None
_connected_ws: Set[WebSocket] = set()
_engine_task: asyncio.Task | None = None
_replay_data: dict[str, Any] | None = None  # set in replay mode


def set_engine(engine: Any) -> None:
    """Inject a SimulationEngine and wire up the broadcast hook."""
    global _engine
    _engine = engine
    engine._dashboard_broadcast = _sync_broadcast


def set_replay_data(data: dict[str, Any]) -> None:
    """Set replay data for Mode 3."""
    global _replay_data
    _replay_data = data


def _sync_broadcast(data: dict[str, Any]) -> None:
    """Synchronous broadcast wrapper for the engine's callback."""
    msg = json.dumps({"type": "tick", **data}, default=str)
    dead = set()
    for ws in _connected_ws:
        try:
            asyncio.get_event_loop().create_task(ws.send_text(msg))
        except Exception:
            dead.add(ws)
    _connected_ws.difference_update(dead)


async def _broadcast_to_all(data: dict[str, Any]) -> None:
    """Async broadcast to all connected WebSocket clients."""
    if not _connected_ws:
        return
    msg = json.dumps(data, default=str)
    dead = set()
    for ws in _connected_ws:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _connected_ws.difference_update(dead)


# -- HTTP endpoints ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse("<h1>Dashboard loading...</h1>")


@app.get("/api/state")
async def get_state():
    if _replay_data is not None:
        return JSONResponse(_replay_data.get("current_state", {}))
    if not _engine:
        return JSONResponse({"error": "no engine"}, status_code=503)
    return JSONResponse(_build_engine_state())


@app.get("/api/protocols")
async def get_protocols():
    return JSONResponse({"protocols": list_protocols()})


@app.get("/api/messages")
async def get_messages(limit: int = 100):
    if not _engine:
        return JSONResponse({"messages": []})
    # Read from events log
    from sanctuary.events import read_events
    events_path = _engine.run_dir.run_dir / "events.jsonl"
    if events_path.exists():
        events = read_events(events_path)
        msgs = [
            e for e in events
            if e.get("event_type") == "message_sent"
        ][-limit:]
        return JSONResponse({"messages": msgs})
    return JSONResponse({"messages": []})


@app.get("/api/analytics")
async def get_analytics():
    if not _engine:
        return JSONResponse({})
    return JSONResponse({
        "total_tactical_calls": _engine.total_tactical_calls,
        "total_strategic_calls": _engine.total_strategic_calls,
        "parse_failures": _engine.parse_failures,
        "parse_recoveries": _engine.parse_recoveries,
        "current_day": _engine.current_day,
    })


# -- WebSocket -----------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_ws.add(websocket)

    # Send initial state
    if _engine:
        try:
            init_data = {"type": "init", **_build_engine_state()}
            await websocket.send_text(json.dumps(init_data, default=str))
        except Exception:
            pass
    elif _replay_data:
        try:
            init_data = {"type": "init", **_replay_data.get("current_state", {})}
            await websocket.send_text(json.dumps(init_data, default=str))
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            await _handle_ws_command(msg, websocket)
    except WebSocketDisconnect:
        _connected_ws.discard(websocket)
    except Exception:
        _connected_ws.discard(websocket)


async def _handle_ws_command(msg: dict[str, Any], ws: WebSocket) -> None:
    """Handle a WebSocket command from the client."""
    cmd = msg.get("cmd")
    if not _engine:
        # Replay mode commands
        if _replay_data and cmd == "scrub":
            day = int(msg.get("day", 1))
            state = _get_replay_state_at_day(day)
            if state:
                await ws.send_text(json.dumps({"type": "tick", **state}, default=str))
        return

    if cmd == "pause":
        _engine._paused = True
    elif cmd == "resume":
        _engine._paused = False
    elif cmd == "set_speed":
        _engine._tick_speed = float(msg.get("seconds", 1.0))
    elif cmd == "fast_forward":
        _engine._fast_forward = bool(msg.get("enabled", True))
    elif cmd == "get_agent":
        agent_id = msg.get("agent_id")
        if agent_id and agent_id in _engine.agents:
            agent = _engine.agents[agent_id]
            detail = {
                "type": "agent_detail",
                "agent_id": agent_id,
                "name": agent.name,
                "role": agent.role,
                "tactical_history": agent.tactical_history[-20:],
                "strategic_history": agent.strategic_history[-10:],
                "policy_history": [
                    {"day": p.day, "week": p.week, "memo": p.raw_memo, "policy": p.policy_json}
                    for p in agent.policy_history
                ],
                "current_policy": (
                    {"day": agent.current_policy.day, "memo": agent.current_policy.raw_memo}
                    if agent.current_policy else None
                ),
                "tactical_call_count": agent.tactical_call_count,
                "strategic_call_count": agent.strategic_call_count,
            }
            try:
                await ws.send_text(json.dumps(detail, default=str))
            except Exception:
                pass


# -- Helpers -------------------------------------------------------------------

def _build_engine_state() -> dict[str, Any]:
    """Build current state dict from engine for API/WebSocket."""
    if not _engine:
        return {}

    snapshot = _engine.market.daily_snapshot()
    return {
        "day": _engine.current_day,
        "max_days": _engine.config.run.days,
        "protocol": _engine.protocol.name,
        "paused": _engine._paused,
        "fast_forward": _engine._fast_forward,
        "completed": _engine.current_day >= _engine.config.run.days,
        **snapshot,
        "stats": {
            "total_tactical_calls": _engine.total_tactical_calls,
            "total_strategic_calls": _engine.total_strategic_calls,
            "total_transactions": len(_engine.market.transactions),
            "parse_failures": _engine.parse_failures,
        },
    }


def _get_replay_state_at_day(day: int) -> dict[str, Any] | None:
    """Get state snapshot for a specific day in replay mode."""
    if not _replay_data:
        return None
    snapshots = _replay_data.get("daily_snapshots", [])
    for snap in snapshots:
        if snap.get("day") == day:
            return snap
    return None
