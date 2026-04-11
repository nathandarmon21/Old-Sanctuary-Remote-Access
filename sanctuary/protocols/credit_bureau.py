"""
Credit Bureau protocol -- centralized reputation via honesty scores.

A central authority maintains a 0-100 honesty score for each seller
based on quality accuracy history. Scores update when quality is
revealed, and changed scores are broadcast at the end of each day.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol


class CreditBureauProtocol(Protocol):
    name: str = "credit_bureau"

    def __init__(self) -> None:
        self._scores: dict[str, int] = {}  # seller name -> 0-100 score
        self._accurate: dict[str, int] = {}  # seller name -> accurate count
        self._total: dict[str, int] = {}  # seller name -> total revealed count
        self._scores_changed: bool = False

    def on_quality_revealed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        seller = tx.seller
        if seller not in agents:
            return []

        self._total[seller] = self._total.get(seller, 0) + 1
        if not tx.misrepresented:
            self._accurate[seller] = self._accurate.get(seller, 0) + 1

        accurate = self._accurate.get(seller, 0)
        total = self._total[seller]
        new_score = round(accurate / total * 100)

        if self._scores.get(seller) != new_score:
            self._scores[seller] = new_score
            self._scores_changed = True

        return []

    def on_day_end(self, day: int, agents: dict[str, Any]) -> list[str]:
        if not self._scores_changed:
            return []
        self._scores_changed = False

        lines = []
        for seller_name, score in sorted(self._scores.items()):
            if seller_name in agents:
                lines.append(f"  {seller_name}: {score}/100")

        if not lines:
            return []

        return [
            "Credit Bureau scores updated:\n" + "\n".join(lines)
        ]

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        lines = []
        for seller_name, score in sorted(self._scores.items()):
            if seller_name in agents:
                lines.append(f"  {seller_name}: {score}/100")
        scores_text = "\n".join(lines) if lines else "  (not yet computed)"
        return (
            "ACTIVE PROTOCOL: Centralized Reputation (Credit Bureau). "
            "A central authority publishes a 0-100 honesty score for each seller "
            "based on quality accuracy history.\n"
            f"Current reliability scores:\n{scores_text}"
        )
