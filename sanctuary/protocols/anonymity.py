"""
Anonymity protocol -- buyer identities hidden from sellers.

All buyer identities are replaced with anonymous labels (Buyer-A, Buyer-B,
etc.) in seller-facing contexts. Private messaging between agents is
disabled. Prevents targeted exploitation based on known quota pressure
or transaction history.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class AnonymityProtocol(Protocol):
    name: str = "anonymity"
    disables_messaging: bool = True
    strips_seller_identity: bool = True

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        return (
            "ACTIVE PROTOCOL: Full Anonymity. "
            "Buyer identities are hidden from sellers. "
            "You cannot identify other agents by name. "
            "No private messaging is allowed. "
            "Every transaction is with an anonymous counterparty. "
            "Buyer transaction history does NOT record seller names."
        )
