"""
Quality revelation scheduling for the Sanctuary simulation.

After a transaction, the true quality of the traded widgets is revealed
publicly to all agents after a deterministic 5-day lag.

This is common knowledge: all agents know quality is revealed on
transaction_day + 5. The specific lag is fixed, not stochastic.
"""

from __future__ import annotations

from dataclasses import dataclass

from sanctuary.economics import REVELATION_LAG_DAYS


# -- Data classes --------------------------------------------------------------

@dataclass(frozen=True)
class PendingRevelation:
    """Internal queue entry. Not visible to simulation agents."""
    transaction_id: str
    seller: str
    buyer: str
    claimed_quality: str
    true_quality: str
    quantity: int
    transaction_day: int
    revelation_day: int  # the day this fires


@dataclass(frozen=True)
class RevelationEvent:
    """
    Public event broadcast to all agents on revelation_day.
    Contains both claimed and true quality so agents can update
    their beliefs about counterparty trustworthiness.
    """
    transaction_id: str
    seller: str
    buyer: str
    claimed_quality: str
    true_quality: str
    quantity: int
    transaction_day: int
    revelation_day: int

    @property
    def misrepresented(self) -> bool:
        """True if the seller misrepresented the quality."""
        return self.claimed_quality != self.true_quality


# -- Scheduler -----------------------------------------------------------------

class RevelationScheduler:
    """
    Maintains the pending revelation queue and fires events on the correct day.

    Uses a deterministic 5-day lag: revelation_day = transaction_day + 5.
    No RNG needed.
    """

    def __init__(self) -> None:
        self._pending: list[PendingRevelation] = []

    def schedule(
        self,
        transaction_id: str,
        seller: str,
        buyer: str,
        claimed_quality: str,
        true_quality: str,
        quantity: int,
        transaction_day: int,
    ) -> int:
        """
        Schedule a revelation for a newly completed transaction.

        Returns the revelation day (transaction_day + 5).
        """
        revelation_day = transaction_day + REVELATION_LAG_DAYS
        self._pending.append(
            PendingRevelation(
                transaction_id=transaction_id,
                seller=seller,
                buyer=buyer,
                claimed_quality=claimed_quality,
                true_quality=true_quality,
                quantity=quantity,
                transaction_day=transaction_day,
                revelation_day=revelation_day,
            )
        )
        return revelation_day

    def fire(self, day: int) -> list[RevelationEvent]:
        """
        Return all revelations due on or before `day` and remove them
        from the pending queue.
        """
        due: list[RevelationEvent] = []
        remaining: list[PendingRevelation] = []

        for pending in self._pending:
            if pending.revelation_day <= day:
                due.append(
                    RevelationEvent(
                        transaction_id=pending.transaction_id,
                        seller=pending.seller,
                        buyer=pending.buyer,
                        claimed_quality=pending.claimed_quality,
                        true_quality=pending.true_quality,
                        quantity=pending.quantity,
                        transaction_day=pending.transaction_day,
                        revelation_day=pending.revelation_day,
                    )
                )
            else:
                remaining.append(pending)

        self._pending = remaining
        return due

    def pending_count(self) -> int:
        """Number of revelations still in the queue."""
        return len(self._pending)

    def all_pending(self) -> list[PendingRevelation]:
        """Read-only view of the pending queue (for logging/checkpointing)."""
        return list(self._pending)
