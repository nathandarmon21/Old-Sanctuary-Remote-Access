"""Tests for sanctuary/messaging.py.

Covers MessageRouter routing semantics: public vs private, sender exclusion
on public broadcasts, sub-round / round-tagged filtering, and the new
deliveries_for_round helper used by the multi-round engine.
"""

from __future__ import annotations

from sanctuary.messaging import MessageRouter, InactivityTracker


class TestMessageRouterDeliveriesForRound:
    def test_private_message_routes_only_to_recipient(self):
        router = MessageRouter(day=1)
        router.send(sender="A", recipient="B", content="hi", is_public=False, sub_round=1)
        deliveries = router.deliveries_for_round(round_num=1, agent_names=["A", "B", "C"])
        assert deliveries["A"] == []  # sender excluded
        assert deliveries["B"] == [{"from": "A", "body": "hi", "public": False}]
        assert deliveries["C"] == []  # not the recipient

    def test_public_message_broadcasts_to_all_except_sender(self):
        router = MessageRouter(day=1)
        router.send(sender="A", recipient="all", content="ann", is_public=True, sub_round=1)
        deliveries = router.deliveries_for_round(round_num=1, agent_names=["A", "B", "C"])
        assert deliveries["A"] == []
        assert deliveries["B"] == [{"from": "A", "body": "ann", "public": True}]
        assert deliveries["C"] == [{"from": "A", "body": "ann", "public": True}]

    def test_round_filter_excludes_other_rounds(self):
        router = MessageRouter(day=1)
        router.send(sender="A", recipient="B", content="r1", is_public=False, sub_round=1)
        router.send(sender="A", recipient="B", content="r2", is_public=False, sub_round=2)
        d1 = router.deliveries_for_round(round_num=1, agent_names=["A", "B"])
        d2 = router.deliveries_for_round(round_num=2, agent_names=["A", "B"])
        assert d1["B"] == [{"from": "A", "body": "r1", "public": False}]
        assert d2["B"] == [{"from": "A", "body": "r2", "public": False}]

    def test_unknown_recipient_dropped(self):
        """If a private message is addressed to an agent not in the
        delivery list (e.g., bankrupt or unknown), it is silently
        dropped — no key error, no spurious delivery."""
        router = MessageRouter(day=1)
        router.send(sender="A", recipient="Z", content="x", is_public=False, sub_round=1)
        deliveries = router.deliveries_for_round(round_num=1, agent_names=["A", "B"])
        assert deliveries["A"] == []
        assert deliveries["B"] == []

    def test_empty_round_returns_empty_lists_per_agent(self):
        router = MessageRouter(day=1)
        deliveries = router.deliveries_for_round(round_num=1, agent_names=["A", "B"])
        assert deliveries == {"A": [], "B": []}

    def test_multiple_messages_preserve_send_order(self):
        router = MessageRouter(day=1)
        router.send(sender="A", recipient="B", content="first", is_public=False, sub_round=1)
        router.send(sender="C", recipient="B", content="second", is_public=False, sub_round=1)
        deliveries = router.deliveries_for_round(round_num=1, agent_names=["A", "B", "C"])
        assert [m["body"] for m in deliveries["B"]] == ["first", "second"]


class TestInactivityTracker:
    """Light coverage of InactivityTracker — exercised indirectly by
    test_engine, but worth a direct check on the tracker's state machine."""

    def test_active_resets_counter(self):
        tracker = InactivityTracker(["A", "B"], threshold=2)
        tracker.advance_day()  # A and B inactive: counters now 1
        assert tracker.consecutive_inactive_days("A") == 1
        tracker.mark_active("A")
        tracker.advance_day()  # A reset, B now 2
        assert tracker.consecutive_inactive_days("A") == 0
        assert tracker.consecutive_inactive_days("B") == 2

    def test_threshold_returns_nudge_targets(self):
        tracker = InactivityTracker(["A"], threshold=2)
        targets1 = tracker.advance_day()  # 1
        targets2 = tracker.advance_day()  # 2
        assert targets1 == []
        assert "A" in targets2
