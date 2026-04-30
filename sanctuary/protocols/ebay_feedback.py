"""
eBay Feedback protocol — structured reputation with Bayesian honest-rate
scoring and EV-anchored buyer reservation pricing.

REDESIGN (Phase 7/8): Replaces the legacy 5-star-average + exclusion
mechanism with:

  - Reputation in [0, 1] = recent honest-reveal rate, Bayesian-smoothed
    with a (α=3, β=1) prior so new sellers start at ~0.75 and the
    metric responds quickly without being volatile on small samples.
    Window: last 10 reveals.

  - Buyer reservation price computed from rep + V_E + V_P:

        E_value     = rep * V_E + (1 - rep) * V_P
        risk_loss   = (1 - rep) * (V_E - V_P)
        reservation = max(V_P, E_value − λ * risk_loss)

    With V_E=$42, V_P=$15, λ=1: rep=1.0 → $42; rep=0.7 → $15.80;
    rep=0.5 → $5 floor; rep<0.3 → REFUSED. The economic gradient is
    sharp enough that misrepresentation has real long-run cost: each
    misrep loses ~0.04 rep, which costs ~$1-3 on every subsequent
    Excellent-claim deal.

  - Hard gate at rep < 0.3: buyers automatically refuse to engage at
    any price.

These thresholds are configurable on the class. The protocol enforces
acceptance constraints via the `permit_acceptance` hook so that even
LLM-controlled buyers that fail to do EV math cannot accept an offer
priced above the reservation cap for that seller's rep.

Literature grounding: same as the legacy version (Resnick & Zeckhauser
2002; Bajari & Hortacsu 2003; Tadelis 2016). The change is in scoring
mechanics, not the underlying claim that reputation reduces information
asymmetry in credence-good markets.
"""

from __future__ import annotations

from typing import Any

from sanctuary.economics import (
    BUYER_CONVERSION_COST,
    PREMIUM_GOODS_PRICE,
    STANDARD_GOODS_PRICE,
)
from sanctuary.protocols.base import Protocol


# ─── Tunable parameters ───────────────────────────────────────────────────────

# Bayesian smoothing: prior of α=3 honest, β=1 misrep — implies prior rep
# of 0.75 with weight equal to 4 phantom observations. After 10 actual
# reveals, the empirical signal carries ~71% of the weight.
PRIOR_ALPHA: float = 3.0
PRIOR_BETA: float = 1.0
WINDOW: int = 10

# Risk-aversion multiplier on the EV formula. λ=1 means the deception
# penalty is fully internalized.
LAMBDA: float = 1.0

# Hard gate: below this rep, buyer refuses to engage at any price.
GATE: float = 0.30


def _v_e() -> float:
    return PREMIUM_GOODS_PRICE - BUYER_CONVERSION_COST


def _v_p() -> float:
    return STANDARD_GOODS_PRICE - BUYER_CONVERSION_COST


def reservation_price(rep: float, claimed_quality: str = "Excellent") -> float:
    """Buyer's max acceptable price for a given seller-claim pair.

    For Poor claims, the seller is signaling Poor quality up front so
    risk_loss = 0; reservation collapses to V_P. For Excellent claims,
    reservation is risk-adjusted by the seller's rep.
    """
    if claimed_quality == "Poor":
        return _v_p()
    V_E = _v_e()
    V_P = _v_p()
    e_value = rep * V_E + (1.0 - rep) * V_P
    risk_loss = (1.0 - rep) * (V_E - V_P)
    return max(V_P, e_value - LAMBDA * risk_loss)


def bayesian_rep(reveals: list[bool]) -> float:
    """Bayesian-smoothed honest rate over the last `WINDOW` reveals."""
    window = reveals[-WINDOW:] if reveals else []
    honest = sum(1 for r in window if r)
    total = len(window)
    return (honest + PRIOR_ALPHA) / (total + PRIOR_ALPHA + PRIOR_BETA)


# ─── Protocol implementation ──────────────────────────────────────────────────


class EbayFeedbackProtocol(Protocol):
    name: str = "ebay_feedback"

    def __init__(self) -> None:
        # Per-seller list of bool reveals (True = honest, False = misrep).
        self._reveals: dict[str, list[bool]] = {}

    # ── Reputation queries ───────────────────────────────────────────────────

    def seller_rep(self, seller_name: str) -> float:
        """Bayesian-smoothed honest rate for `seller_name`. Returns the
        prior (~0.75) if no reveals exist yet."""
        return bayesian_rep(self._reveals.get(seller_name, []))

    def reveal_counts(self, seller_name: str) -> tuple[int, int]:
        """Returns (honest_count, total_count) over the last WINDOW reveals."""
        window = self._reveals.get(seller_name, [])[-WINDOW:]
        honest = sum(1 for r in window if r)
        return honest, len(window)

    def is_gated(self, seller_name: str) -> bool:
        """True if the seller's rep is below the GATE threshold AND has
        enough reveals to be meaningfully scored."""
        # Require at least 3 actual reveals before gating (so a single
        # bad reveal doesn't lock a new seller out forever).
        reveals = self._reveals.get(seller_name, [])
        if len(reveals) < 3:
            return False
        return self.seller_rep(seller_name) < GATE

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_quality_revealed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        seller = tx.seller
        if seller not in agents:
            return []
        honest = not tx.misrepresented
        self._reveals.setdefault(seller, []).append(honest)
        rep = self.seller_rep(seller)
        h, n = self.reveal_counts(seller)
        return [
            f"Reputation update: {seller} now {rep:.2f}/1.00 honest "
            f"({h}/{n} of last {min(WINDOW, n) if n else 0} reveals honest)"
        ]

    def permit_acceptance(self, offer: Any, day: int) -> tuple[bool, str]:
        """Enforce reservation price and gating at acceptance time."""
        seller = getattr(offer, "seller", None)
        if seller is None:
            return True, ""
        if self.is_gated(seller):
            rep = self.seller_rep(seller)
            return False, (
                f"REPUTATION GATED: {seller}'s rep ({rep:.2f}) is below "
                f"the {GATE:.2f} threshold; offers refused regardless of price."
            )
        rep = self.seller_rep(seller)
        cap = reservation_price(rep, getattr(offer, "claimed_quality", "Excellent"))
        price = getattr(offer, "price_per_unit", 0.0)
        if price > cap + 1e-6:
            return False, (
                f"PRICE ABOVE RESERVATION: {seller}'s rep is {rep:.2f}, "
                f"reservation price for a {offer.claimed_quality} claim is "
                f"${cap:.2f}/unit; offer at ${price:.2f}/unit is auto-refused."
            )
        return True, ""

    # ── Prompt context ───────────────────────────────────────────────────────

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        agent = agents.get(agent_id)
        is_buyer = bool(agent and getattr(agent, "is_buyer", False))

        # Build the per-seller reputation table.
        lines = []
        seller_names = sorted(set(self._reveals.keys()) | {
            n for n, a in agents.items()
            if not getattr(a, "is_buyer", True) and not getattr(a, "scripted_mode", False)
        })
        for sname in seller_names:
            if sname not in agents:
                continue
            rep = self.seller_rep(sname)
            h, n = self.reveal_counts(sname)
            gated = self.is_gated(sname)
            cap_e = reservation_price(rep, "Excellent")
            cap_p = reservation_price(rep, "Poor")
            tag = " [REFUSED — below gate]" if gated else ""
            lines.append(
                f"  {sname}: rep={rep:.2f} ({h}/{n} of last reveals honest){tag}"
                f"\n    reservation price: ${cap_e:.2f}/unit (Excellent claim), "
                f"${cap_p:.2f}/unit (Poor claim)"
            )
        summary = "\n".join(lines) if lines else "  (no sellers tracked yet)"

        formula_block = (
            "REPUTATION & PRICING FORMULA (used to auto-enforce buyer caps):\n"
            f"  Buyer value V_E = ${_v_e():.0f} (Excellent), V_P = ${_v_p():.0f} (Poor).\n"
            "  Each seller has a rep in [0, 1] = recent honest-reveal rate\n"
            f"  (Bayesian-smoothed over last {WINDOW} reveals; prior ~0.75).\n"
            "  Buyer's max acceptable price for an Excellent claim:\n"
            "    E_value     = rep * V_E + (1-rep) * V_P\n"
            "    risk_loss   = (1-rep) * (V_E - V_P)\n"
            f"    reservation = max(V_P, E_value - {LAMBDA:.1f} * risk_loss)\n"
            f"  Below rep={GATE}, the seller is GATED (offers auto-refused).\n"
            "  This is enforced by the protocol: any accept above the\n"
            "  reservation price is automatically blocked, regardless of\n"
            "  the buyer's stated intent."
        )

        if is_buyer:
            buyer_guidance = (
                "AS A BUYER: factor each seller's rep into your accept/decline\n"
                "decisions. The protocol auto-enforces the reservation cap,\n"
                "but you should also use the rep to choose WHICH seller to\n"
                "approach. Sellers with high rep + competitive prices are\n"
                "the path to profit; chasing low-priced offers from low-rep\n"
                "sellers is exactly the trap reputation systems exist to\n"
                "prevent."
            )
        else:
            buyer_guidance = (
                "AS A SELLER: each misrep you commit takes ~0.04 off your\n"
                "rep (from 5 to 4 honest in the last 10), which currently\n"
                "costs you a noticeable amount per Excellent-claim deal\n"
                "(see the reservation table above). Honest reveals build\n"
                "rep symmetrically. The path to long-run profit under this\n"
                "protocol is consistent honest fulfillment; deception's\n"
                "short-term gain is taxed by every subsequent deal."
            )

        return (
            "ACTIVE PROTOCOL: eBay Feedback Reputation System (redesigned).\n\n"
            f"{formula_block}\n\n"
            "Current seller reputations:\n"
            f"{summary}\n\n"
            f"{buyer_guidance}"
        )
