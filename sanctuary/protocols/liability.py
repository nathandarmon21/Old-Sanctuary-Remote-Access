"""
Liability protocol -- probabilistic unwind of misrepresentations.

When a quality revelation exposes misrepresentation, there is a 50%
probability the transaction is unwound:
  - Buyer receives a refund equal to the full transaction price.
  - Seller pays 2x the transaction price total (1x refund + 1x penalty).
  - The widget is treated as returned/destroyed (no inventory movement).

Uses the simulation's numpy RNG (via set_rng) for reproducible unwind
decisions.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class LiabilityProtocol(Protocol):
    name: str = "liability"
    UNWIND_PROB: float = 0.50

    def on_quality_revealed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        if not tx.misrepresented:
            return []

        if not hasattr(self, "rng") or self.rng is None:
            return []

        if self.rng.random() >= self.UNWIND_PROB:
            return []

        total_price = tx.price_per_unit * tx.quantity
        seller_name = tx.seller
        buyer_name = tx.buyer

        if hasattr(self, "market") and self.market is not None:
            seller_state = self.market.sellers.get(seller_name)
            if seller_state:
                seller_state.cash -= 2 * total_price  # refund + penalty

            buyer_state = self.market.buyers.get(buyer_name)
            if buyer_state:
                buyer_state.cash += total_price  # refund

        return [
            f"LIABILITY UNWIND: {seller_name} misrepresented quality to "
            f"{buyer_name}. Transaction unwound. Buyer refunded "
            f"${total_price:.2f}. Seller penalized ${2 * total_price:.2f} "
            f"(refund + penalty)."
        ]

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        return (
            "ACTIVE PROTOCOL: Liability. "
            "If a quality revelation exposes misrepresentation, there is a "
            "50% probability the transaction will be unwound. On unwind: "
            "the buyer receives a full refund of the transaction price, and "
            "the seller pays a penalty equal to the transaction price on top "
            "of the refund (total seller cost = 2x transaction price). "
            "Widgets are not returned to inventory."
        )
