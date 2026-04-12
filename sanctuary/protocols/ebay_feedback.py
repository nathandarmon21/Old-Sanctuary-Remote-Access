"""
eBay Feedback protocol -- structured quantitative reputation via buyer ratings.

After each quality revelation, the affected buyer posts a public 1-5 star
rating for the seller. Accurate quality claims receive 5 stars;
misrepresentations receive 1 star. All agents see the full rating
history (running average and count) for every seller.

Literature grounding:
  This protocol implements the conservative baseline reputation mechanism
  studied extensively in the experimental economics literature on online
  marketplaces. Key references:

  - Resnick, P. and Zeckhauser, R. (2002). "Trust Among Strangers in
    Internet Transactions: Empirical Analysis of eBay's Reputation System."
    Advances in Applied Microeconomics, 11.

  - Bajari, P. and Hortacsu, A. (2003). "The Winner's Curse, Reserve Prices,
    and Endogenous Entry: Empirical Insights from eBay Auctions."
    RAND Journal of Economics, 34(2).

  - Tadelis, S. (2016). "Reputation and Feedback Systems in Online Platform
    Markets." Annual Review of Economics, 8.

  These studies establish that simple star-rating systems significantly reduce
  information asymmetry in credence-good markets but are vulnerable to
  strategic manipulation (retaliatory ratings, rating inflation). The
  Sanctuary simulation tests whether LLM agents exhibit similar patterns.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class EbayFeedbackProtocol(Protocol):
    name: str = "ebay_feedback"

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
            f"eBay-style feedback updated: {seller} now {avg:.1f}/5 stars "
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
            "ACTIVE PROTOCOL: eBay Feedback. "
            "Buyers publicly rate sellers 1-5 stars after each quality revelation. "
            "All agents can see these ratings.\n"
            f"Current seller ratings:\n{summary}"
        )
