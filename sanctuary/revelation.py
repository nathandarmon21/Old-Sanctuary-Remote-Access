"""
Quality revelation scheduling for the Sanctuary simulation.

After a transaction, the true quality of the traded widgets is revealed
publicly to all agents after a stochastic delay. The delay distribution is:

    Delay (days):  1     2     3     4     5     6
    Probability:   0.05  0.15  0.20  0.30  0.20  0.10

Properties:
  - Sum = 1.00
  - Mode = 4 days
  - Median = 4 days (CDF reaches 0.70 at day 4)

This distribution is common knowledge. The specific realization for any
given transaction is sampled at transaction time but not visible to agents
until the revelation day arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Revelation delay distribution ────────────────────────────────────────────

REVELATION_DELAYS: list[int] = [1, 2, 3, 4, 5, 6]
REVELATION_PMF: list[float] = [0.05, 0.15, 0.20, 0.30, 0.20, 0.10]

assert abs(sum(REVELATION_PMF) - 1.0) < 1e-9, "Revelation PMF must sum to 1.0"
assert len(REVELATION_DELAYS) == len(REVELATION_PMF), "PMF length mismatch"

_DELAYS_ARRAY = np.array(REVELATION_DELAYS)
_PMF_ARRAY = np.array(REVELATION_PMF)


# ── Data classes ──────────────────────────────────────────────────────────────

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


# ── Scheduler ────────────────────────────────────────────────────────────────

class RevelationScheduler:
    """
    Maintains the pending revelation queue and fires events on the correct day.

    Designed to be driven by a single master RNG so that the entire
    revelation schedule is reproducible from a seed.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
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

        Samples the delay from the PMF using the master RNG, records the
        pending revelation, and returns the sampled revelation day so it can
        be written into the transaction log.
        """
        delay = self._sample_delay()
        revelation_day = transaction_day + delay
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

        In normal operation this is called once per simulated day with
        `day = current_day`. The `<= day` guard handles any edge cases
        where a revelation day was missed (e.g., agent bankruptcy removed
        them from the day loop but revelations still propagate).
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
        """Read-only view of the pending queue (for logging)."""
        return list(self._pending)

    def _sample_delay(self) -> int:
        return int(self._rng.choice(_DELAYS_ARRAY, p=_PMF_ARRAY))
