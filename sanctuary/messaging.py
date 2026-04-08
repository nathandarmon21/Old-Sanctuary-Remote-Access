"""
Message routing and offer ledger for the Sanctuary simulation.

Manages:
  - Agent messages (private point-to-point and public broadcasts)
  - The pending offer ledger (separate from MarketState so message routing
    can be tested independently of market economics)
  - Inactivity tracking (consecutive days without any action)

Note on offer placement: MarketState.place_offer() is the authority for
offer validation and state mutation. This module provides the routing
plumbing that sits above it — parsing action payloads, dispatching to
MarketState, and collecting the resulting messages/offers for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message sent by one agent during a simulation day."""
    message_id: str
    sender: str
    recipient: str        # agent name, or "all" for broadcasts
    is_public: bool
    day: int
    sub_round: int        # 0 = main tactical pass, 1-2 = sub-rounds
    content: str


@dataclass
class MessageRouter:
    """
    Collects messages produced during a single simulation day and
    provides views per recipient for prompt construction.

    A new MessageRouter is created each day; the simulation loop
    archives the day's messages to the JSONL log before discarding it.
    """

    day: int
    _messages: list[Message] = field(default_factory=list)
    _next_id: int = 0

    def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        is_public: bool,
        sub_round: int = 0,
    ) -> Message:
        """Record a message. Returns the stored Message object."""
        msg = Message(
            message_id=f"msg_{self.day}_{self._next_id}",
            sender=sender,
            recipient=recipient,
            is_public=is_public,
            day=self.day,
            sub_round=sub_round,
            content=content,
        )
        self._messages.append(msg)
        self._next_id += 1
        return msg

    def messages_for(self, agent_name: str, sub_round: int | None = None) -> list[Message]:
        """
        Return messages visible to agent_name: public messages plus
        private messages addressed to them.

        If sub_round is specified, only messages from that sub-round
        are returned. Otherwise all messages up to and including the
        main pass (sub_round=0) are returned for the sub-round context.
        """
        result = []
        for msg in self._messages:
            if sub_round is not None and msg.sub_round != sub_round:
                continue
            if msg.is_public or msg.recipient == agent_name:
                result.append(msg)
        return result

    def all_messages(self) -> list[Message]:
        """All messages this day, in send order."""
        return list(self._messages)

    def as_log_records(self) -> list[dict]:
        """Serialise all messages for JSONL logging."""
        return [
            {
                "message_id": m.message_id,
                "sender": m.sender,
                "recipient": m.recipient,
                "is_public": m.is_public,
                "day": m.day,
                "sub_round": m.sub_round,
                "content": m.content,
            }
            for m in self._messages
        ]


# ── Inactivity tracker ────────────────────────────────────────────────────────

class InactivityTracker:
    """
    Tracks consecutive inactive days per agent.

    An agent is considered active on day D if it sent at least one
    message, made at least one offer, accepted or declined at least
    one offer, or produced widgets / final goods.

    Agents inactive for >= threshold consecutive days receive a nudge
    prepended to their next tactical call.
    """

    def __init__(self, agent_names: list[str], threshold: int = 2) -> None:
        self.threshold = threshold
        self._consecutive: dict[str, int] = {name: 0 for name in agent_names}
        self._active_today: set[str] = set()

    def mark_active(self, agent_name: str) -> None:
        """Call whenever an agent takes any action."""
        self._active_today.add(agent_name)

    def advance_day(self) -> list[str]:
        """
        End-of-day update. Returns names of agents who have been inactive
        for >= threshold consecutive days (including today).
        """
        nudge_targets: list[str] = []
        for name in self._consecutive:
            if name in self._active_today:
                self._consecutive[name] = 0
            else:
                self._consecutive[name] += 1
                if self._consecutive[name] >= self.threshold:
                    nudge_targets.append(name)
        self._active_today = set()
        return nudge_targets

    def consecutive_inactive_days(self, agent_name: str) -> int:
        return self._consecutive.get(agent_name, 0)

    def nudge_text(self, agent_name: str) -> str:
        days = self._consecutive[agent_name]
        return (
            f"[PARTICIPATION NOTICE] {agent_name} has not taken any market action "
            f"for {days} consecutive day(s). Your fixed costs are accruing. "
            f"Consider trading, producing, or sending messages today."
        )
