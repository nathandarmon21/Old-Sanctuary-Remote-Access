"""Tests for the redesigned reputation EV formula + auto-discount/gating
(redesign 7/8).

Verifies the math of `reservation_price` and `bayesian_rep`, plus the
`permit_acceptance` hook that enforces price caps and gating at the
engine acceptance path.
"""

from __future__ import annotations

import pytest

from sanctuary.protocols.ebay_feedback import (
    EbayFeedbackProtocol,
    GATE,
    LAMBDA,
    bayesian_rep,
    reservation_price,
)


class _FakeOffer:
    def __init__(self, seller, claimed_quality, price_per_unit):
        self.seller = seller
        self.claimed_quality = claimed_quality
        self.price_per_unit = price_per_unit


class _FakeTx:
    def __init__(self, seller, misrepresented):
        self.seller = seller
        self.misrepresented = misrepresented


# ─── bayesian_rep ─────────────────────────────────────────────────────────────


class TestBayesianRep:
    def test_empty_returns_prior(self):
        assert bayesian_rep([]) == pytest.approx(0.75)

    def test_one_honest(self):
        # (1+3)/(1+4) = 0.80
        assert bayesian_rep([True]) == pytest.approx(0.80)

    def test_one_misrep(self):
        # (0+3)/(1+4) = 0.60
        assert bayesian_rep([False]) == pytest.approx(0.60)

    def test_window_caps_at_10(self):
        # 12 reveals, but only last 10 count: 10 honest, 0 misrep
        reveals = [False, False] + [True] * 10
        # Last 10 are all True: (10+3)/(10+4) ≈ 0.929
        assert bayesian_rep(reveals) == pytest.approx(13 / 14)


# ─── reservation_price ────────────────────────────────────────────────────────


class TestReservationPrice:
    def test_perfect_rep_pays_v_e(self):
        # rep=1.0 -> E_value=$42, risk_loss=0, reservation = $42
        assert reservation_price(1.0, "Excellent") == pytest.approx(42.0)

    def test_zero_rep_floors_at_v_p(self):
        # rep=0 -> E_value=15, risk_loss=27, reservation = max(15, 15-27) = $15
        assert reservation_price(0.0, "Excellent") == pytest.approx(15.0)

    def test_mid_rep_quadratic_drop(self):
        # rep=0.7 -> E_value = 0.7*42 + 0.3*15 = 33.9
        # risk_loss = 0.3 * 27 = 8.1
        # reservation = 33.9 - 8.1 = 25.8 (lambda=1)
        assert reservation_price(0.7, "Excellent") == pytest.approx(25.8, abs=0.05)

    def test_poor_claim_collapses_to_v_p(self):
        # Poor claims have no upside, no risk_loss -> reservation = V_P
        for rep in (0.1, 0.5, 1.0):
            assert reservation_price(rep, "Poor") == pytest.approx(15.0)

    def test_rep_below_half_floors_at_v_p(self):
        # rep=0.5 -> E_value = 28.5, risk_loss = 13.5
        # reservation = max(15, 15) = $15 (floor binds)
        assert reservation_price(0.5, "Excellent") == pytest.approx(15.0)


# ─── permit_acceptance hook ───────────────────────────────────────────────────


class TestPermitAcceptance:
    def test_allows_under_cap(self):
        p = EbayFeedbackProtocol()
        # New seller (no reveals) -> prior rep=0.75 -> reservation ≈ $28.50
        offer = _FakeOffer("Aldridge", "Excellent", 25.0)
        ok, reason = p.permit_acceptance(offer, day=1)
        assert ok is True
        assert reason == ""

    def test_blocks_over_cap(self):
        p = EbayFeedbackProtocol()
        offer = _FakeOffer("Aldridge", "Excellent", 35.0)  # > $28.50
        ok, reason = p.permit_acceptance(offer, day=1)
        assert ok is False
        assert "ABOVE RESERVATION" in reason

    def test_gate_below_threshold(self):
        p = EbayFeedbackProtocol()
        # Seed three misreps so seller's rep drops below GATE (0.30).
        agents = {"Aldridge": object()}
        for _ in range(5):
            p.on_quality_revealed(_FakeTx("Aldridge", misrepresented=True), agents)
        # 0 honest / 5 total -> (0+3)/(5+4) = 0.333... still above gate
        # Add one more misrep -> (0+3)/(6+4) = 0.30, exactly at gate (not below)
        # Add yet another -> (0+3)/(7+4) ≈ 0.273, below gate
        for _ in range(2):
            p.on_quality_revealed(_FakeTx("Aldridge", misrepresented=True), agents)
        rep = p.seller_rep("Aldridge")
        assert rep < GATE
        offer = _FakeOffer("Aldridge", "Excellent", 1.0)  # any price
        ok, reason = p.permit_acceptance(offer, day=10)
        assert ok is False
        assert "GATED" in reason

    def test_gate_does_not_fire_on_few_reveals(self):
        """Gating requires at least 3 actual reveals — a single bad reveal
        on a new seller doesn't lock them out."""
        p = EbayFeedbackProtocol()
        agents = {"Aldridge": object()}
        # 1 misrep -> rep = 0.6, but only 1 reveal so gate doesn't fire
        p.on_quality_revealed(_FakeTx("Aldridge", misrepresented=True), agents)
        offer = _FakeOffer("Aldridge", "Excellent", 20.0)  # under cap
        ok, _ = p.permit_acceptance(offer, day=2)
        assert ok is True


# ─── Long-run economic gradient ───────────────────────────────────────────────


class TestSellerSelfBlock:
    """The redesigned get_agent_context surfaces the seller's OWN rep
    plus marginal impact of next-reveal decisions, per redesign 8/8."""

    def test_seller_sees_own_rep_block(self):
        p = EbayFeedbackProtocol()
        # Use a fake "agent" object that the protocol can introspect.
        class _FakeSeller:
            is_seller = True
            is_buyer = False

        agents = {"Aldridge": _FakeSeller()}
        # Two honest reveals -> rep = 5/6 ≈ 0.833
        p.on_quality_revealed(_FakeTx("Aldridge", misrepresented=False), agents)
        p.on_quality_revealed(_FakeTx("Aldridge", misrepresented=False), agents)

        ctx = p.get_agent_context("Aldridge", agents, day=10)
        assert "YOUR REPUTATION" in ctx
        assert "Aldridge (YOU)" in ctx
        assert "Marginal effect of your next reveal" in ctx
        # Honest path should produce a higher rep than misrep path.
        # The exact numbers depend on Bayesian formula: at 2 honest:
        # one more misrep -> 2/3+pseudo = 5/7 ≈ 0.714
        # one more honest -> 3/3+pseudo = 6/7 ≈ 0.857
        assert "0.71" in ctx or "0.72" in ctx
        assert "0.85" in ctx or "0.86" in ctx

    def test_buyer_does_not_see_self_block(self):
        p = EbayFeedbackProtocol()

        class _FakeBuyer:
            is_seller = False
            is_buyer = True

        agents = {"Halcyon": _FakeBuyer()}
        ctx = p.get_agent_context("Halcyon", agents, day=1)
        assert "YOUR REPUTATION" not in ctx
        assert "AS A BUYER" in ctx


class TestEconomicGradient:
    def test_excellent_margin_collapses_with_rep(self):
        """Sanity check: as rep degrades, max acceptable seller price drops
        toward V_P, eroding deception EV."""
        prices = [reservation_price(r, "Excellent") for r in [1.0, 0.9, 0.7, 0.5, 0.3]]
        # Should be monotonically non-increasing
        for a, b in zip(prices, prices[1:]):
            assert a >= b

    def test_lambda_1_means_full_internalization(self):
        # At rep=0.5, with V_E=42, V_P=15: E_value=28.5, risk_loss=13.5
        # Lambda=1 -> reservation = max(15, 15) = $15 (floor)
        # If lambda were 0 (no risk aversion), reservation would be $28.50.
        # The formula uses LAMBDA=1.
        assert LAMBDA == pytest.approx(1.0)
        assert reservation_price(0.5, "Excellent") == pytest.approx(15.0)
