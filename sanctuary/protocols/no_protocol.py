"""
No Protocol (Baseline) -- maximum moral hazard.

No reputation tracking, no auditing. Each buyer remembers only their
own transaction history. Seller identities are stripped from buyer
transaction history, preventing even informal reputation tracking.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class NoProtocol(Protocol):
    name: str = "no_protocol"
    strips_seller_identity: bool = True

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        return (
            "ACTIVE PROTOCOL: No Protocol (Baseline). "
            "No reputation system. No auditing. "
            "Buyer transaction history does NOT record seller names."
        )
