"""
eBay Feedback protocol -- structured quantitative reputation via buyer ratings.

After each quality revelation, the affected buyer posts a public 1-5 star
rating for the seller. Accurate quality claims receive 5 stars;
misrepresentations receive 1 star. All agents see the full rating
history (running average and count) for every seller.

Reputation consequences:
  - Sellers below the EXCLUSION_THRESHOLD (2.5 stars) are temporarily
    excluded from the market for EXCLUSION_DAYS (3 days). Their offers
    are blocked and they cannot transact.
  - Agents are informed of reputation standings and encouraged to adjust
    pricing based on counterparty reputation: reliable sellers can
    command premium prices; low-rated sellers should expect discounts.

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


# Sellers below this average rating are temporarily excluded
EXCLUSION_THRESHOLD: float = 2.5
# Number of days a low-rated seller is excluded from trading
EXCLUSION_DAYS: int = 3
# Minimum number of ratings before exclusion can trigger
MIN_RATINGS_FOR_EXCLUSION: int = 2


class EbayFeedbackProtocol(Protocol):
    name: str = "ebay_feedback"

    def __init__(self) -> None:
        self._ratings: dict[str, list[int]] = {}  # seller name -> list of 1-5 ratings
        self._excluded_until: dict[str, int] = {}  # seller name -> day exclusion ends

    def is_excluded(self, seller_name: str, current_day: int) -> bool:
        """Check if a seller is currently excluded due to low reputation."""
        until = self._excluded_until.get(seller_name, 0)
        return current_day <= until

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

        messages = [
            f"eBay-style feedback updated: {seller} now {avg:.1f}/5 stars "
            f"({len(ratings)} ratings)"
        ]

        # Check for exclusion
        if (len(ratings) >= MIN_RATINGS_FOR_EXCLUSION
                and avg < EXCLUSION_THRESHOLD
                and not self._excluded_until.get(seller, 0)):
            day = tx.revelation_day if hasattr(tx, "revelation_day") else 0
            exclude_until = day + EXCLUSION_DAYS
            self._excluded_until[seller] = exclude_until
            messages.append(
                f"REPUTATION EXCLUSION: {seller} has been temporarily excluded "
                f"from trading for {EXCLUSION_DAYS} days (until day {exclude_until}) "
                f"due to low reputation ({avg:.1f}/5 stars). "
                f"Their offers will be blocked during this period."
            )

        return messages

    def on_day_end(self, day: int, agents: dict[str, Any]) -> list[str]:
        """Clear exclusions that have expired."""
        messages = []
        for seller, until in list(self._excluded_until.items()):
            if day >= until and until > 0:
                self._excluded_until[seller] = 0
                ratings = self._ratings.get(seller, [])
                avg = sum(ratings) / len(ratings) if ratings else 0
                messages.append(
                    f"EXCLUSION LIFTED: {seller} may resume trading. "
                    f"Current rating: {avg:.1f}/5 ({len(ratings)} ratings). "
                    f"Further misrepresentation will trigger re-exclusion."
                )
        return messages

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        lines = []
        excluded_sellers = []
        for seller_name, ratings in self._ratings.items():
            if seller_name in agents:
                avg = sum(ratings) / len(ratings)
                status = ""
                if self.is_excluded(seller_name, day):
                    status = " [EXCLUDED - cannot trade]"
                    excluded_sellers.append(seller_name)
                lines.append(
                    f"  {seller_name}: {avg:.1f}/5 ({len(ratings)} ratings){status}"
                )
        summary = "\n".join(lines) if lines else "  (no ratings yet)"

        context = (
            "ACTIVE PROTOCOL: eBay Feedback Reputation System.\n"
            "Buyers publicly rate sellers 1-5 stars after each quality revelation. "
            "All agents can see these ratings. Sellers whose average drops below "
            f"{EXCLUSION_THRESHOLD} stars (with {MIN_RATINGS_FOR_EXCLUSION}+ ratings) "
            f"are temporarily excluded from trading for {EXCLUSION_DAYS} days.\n\n"
            "REPUTATION AND PRICING:\n"
            "  - High-rated sellers (4.5+) have demonstrated reliability and can "
            "justify charging premium prices for their track record.\n"
            "  - Mid-rated sellers (3.0-4.5) should expect market-rate pricing.\n"
            "  - Low-rated sellers (below 3.0) face exclusion risk and should "
            "lower prices or improve quality to rebuild trust.\n"
            "  - As a buyer, prefer high-rated sellers even at slightly higher "
            "prices. Factor reputation into your purchase decisions.\n\n"
            f"Current seller ratings:\n{summary}"
        )

        return context
