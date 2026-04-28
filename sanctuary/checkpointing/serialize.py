"""JSON serializers/deserializers for engine state objects.

Round-trip: MarketState, Agent (the subset that affects future behavior),
and RevelationScheduler. The resume path uses these to rebuild a fresh
engine into a state observably identical to the run that produced the
checkpoint.

JSON-only — no pickle. Dataclass fields are simple primitives + nested
dicts/lists, which round-trip via `dataclasses.asdict` and explicit
constructors here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sanctuary.market import (
    BuyerState,
    FinalGoodsRecord,
    MarketState,
    PendingOffer,
    SellerState,
    TransactionRecord,
    WidgetInstance,
    WidgetLot,
)
from sanctuary.revelation import PendingRevelation, RevelationScheduler


# ── Market ─────────────────────────────────────────────────────────────────────


def serialize_market(market: MarketState) -> dict[str, Any]:
    return {
        "current_day": market.current_day,
        "fg_prices": dict(market.fg_prices),
        "sellers": {n: dataclasses.asdict(s) for n, s in market.sellers.items()},
        "buyers": {n: dataclasses.asdict(b) for n, b in market.buyers.items()},
        "pending_offers": {
            oid: dataclasses.asdict(o) for oid, o in market.pending_offers.items()
        },
        "transactions": [dataclasses.asdict(t) for t in market.transactions],
    }


def deserialize_market(d: dict[str, Any]) -> MarketState:
    sellers = {n: _seller_from_dict(s) for n, s in d["sellers"].items()}
    buyers = {n: _buyer_from_dict(b) for n, b in d["buyers"].items()}
    market = MarketState(
        sellers=sellers,
        buyers=buyers,
        fg_prices=dict(d.get("fg_prices", {})),
    )
    market.current_day = int(d.get("current_day", 0))
    market.pending_offers = {
        oid: PendingOffer(**o) for oid, o in d.get("pending_offers", {}).items()
    }
    market.transactions = [
        TransactionRecord(**t) for t in d.get("transactions", [])
    ]
    return market


def _seller_from_dict(d: dict[str, Any]) -> SellerState:
    return SellerState(
        name=d["name"],
        cash=float(d["cash"]),
        inventory=dict(d.get("inventory", {})),
        widget_instances=[
            WidgetInstance(**wi) for wi in d.get("widget_instances", [])
        ],
        next_widget_id=int(d.get("next_widget_id", 1)),
        factories=int(d.get("factories", 1)),
        factory_build_queue=list(d.get("factory_build_queue", [])),
        bankrupt=bool(d.get("bankrupt", False)),
        last_active_day=int(d.get("last_active_day", 0)),
        consecutive_inactive_days=int(d.get("consecutive_inactive_days", 0)),
        production_costs_incurred=float(d.get("production_costs_incurred", 0.0)),
        starting_cash=float(d.get("starting_cash", 0.0)),
    )


def _buyer_from_dict(d: dict[str, Any]) -> BuyerState:
    return BuyerState(
        name=d["name"],
        cash=float(d["cash"]),
        widget_lots=[WidgetLot(**wl) for wl in d.get("widget_lots", [])],
        produced_goods_records=[
            FinalGoodsRecord(**fgr) for fgr in d.get("produced_goods_records", [])
        ],
        bankrupt=bool(d.get("bankrupt", False)),
        last_active_day=int(d.get("last_active_day", 0)),
        consecutive_inactive_days=int(d.get("consecutive_inactive_days", 0)),
        widgets_acquired=int(d.get("widgets_acquired", 0)),
        penalties_accrued=float(d.get("penalties_accrued", 0.0)),
        starting_cash=float(d.get("starting_cash", 0.0)),
    )


# ── Agent ──────────────────────────────────────────────────────────────────────


def serialize_agent(agent) -> dict[str, Any]:
    """Serialize the parts of an Agent that influence future behavior.

    Providers, prompt-style knobs, and persona are reconstructed from
    config on resume — only behavioral state needs round-tripping:
    histories, interaction log, current policy, call counters.
    """
    last_policy = agent.current_policy
    return {
        "history": list(agent.history),
        "tactical_history": list(agent.tactical_history),
        "strategic_history": list(agent.strategic_history),
        "interaction_log": list(agent.interaction_log),
        "current_policy": (
            dataclasses.asdict(last_policy) if last_policy is not None else None
        ),
        "policy_history": [dataclasses.asdict(p) for p in agent.policy_history],
        "tactical_call_count": agent.tactical_call_count,
        "strategic_call_count": agent.strategic_call_count,
        "fulfillment_call_count": agent.fulfillment_call_count,
    }


def apply_agent_state(agent, state: dict[str, Any]) -> None:
    """Apply a serialized state dict to an existing Agent in-place."""
    from sanctuary.agent import PolicyRecord
    agent.history = list(state.get("history", []))
    agent.tactical_history = list(state.get("tactical_history", []))
    agent.strategic_history = list(state.get("strategic_history", []))
    agent.interaction_log = list(state.get("interaction_log", []))
    cur = state.get("current_policy")
    agent.current_policy = PolicyRecord(**cur) if cur else None
    agent.policy_history = [
        PolicyRecord(**p) for p in state.get("policy_history", [])
    ]
    agent.tactical_call_count = int(state.get("tactical_call_count", 0))
    agent.strategic_call_count = int(state.get("strategic_call_count", 0))
    agent.fulfillment_call_count = int(state.get("fulfillment_call_count", 0))


# ── Revelation queue ──────────────────────────────────────────────────────────


def serialize_revelation_queue(scheduler: RevelationScheduler) -> list[dict[str, Any]]:
    return [dataclasses.asdict(p) for p in scheduler.all_pending()]


def apply_revelation_queue(
    scheduler: RevelationScheduler, pending: list[dict[str, Any]],
) -> None:
    scheduler._pending = [PendingRevelation(**p) for p in pending]
