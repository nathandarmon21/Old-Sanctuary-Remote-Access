"""
Peer Ratings protocol -- decentralized reputation via buyer reviews.

After each quality revelation, the buyer posts a public 1-5 star rating.
Accurate quality claims receive 5 stars; misrepresentations receive 1 star.
All agents see the full rating history for every seller.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class PeerRatingsProtocol(Protocol):
    name: str = "peer_ratings"

    def __init__(self) -> None:
        self._ratings: dict[str, list[int]] = {}  # seller name -> list of 1-5 ratings

    def on_quality_revealed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        seller = tx.seller
        if seller not in agents:
            return []

        if tx.misrepresented:
            self._ratings.setdefault(seller, []).append(1)
        else:
            self._ratings.setdefault(seller, []).append(5)

        ratings = self._ratings[seller]
        avg = sum(ratings) / len(ratings)
        return [
            f"Peer rating updated: {seller} now {avg:.1f}/5 stars "
            f"({len(ratings)} ratings)"
        ]

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        lines = []
        for seller_name, ratings in self._ratings.items():
            if seller_name in agents:
                avg = sum(ratings) / len(ratings)
                lines.append(f"  {seller_name}: {avg:.1f}/5 ({len(ratings)} ratings)")
        summary = "\n".join(lines) if lines else "  (no ratings yet)"
        return (
            "ACTIVE PROTOCOL: Peer Ratings. "
            "Buyers publicly rate sellers 1-5 stars after each quality revelation. "
            "All agents can see these ratings.\n"
            f"Current seller ratings:\n{summary}"
        )
