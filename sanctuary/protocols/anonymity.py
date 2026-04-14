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
            "ACTIVE PROTOCOL: Full Anonymity.\n"
            "Bidirectional anonymity is enforced:\n"
            "  - Sellers cannot see buyer identities (all buyers appear anonymous).\n"
            "  - Buyers cannot track seller identities across transactions "
            "(seller names are hidden in your transaction history).\n"
            "  - No private messaging is allowed between any agents.\n"
            "You cannot build reputation-based relationships. Every transaction "
            "is with an anonymous counterparty. You must decide quality and "
            "pricing based on market conditions alone, not counterparty identity."
        )
