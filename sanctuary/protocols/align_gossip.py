"""
ALIGN Gossip protocol -- decentralized linguistic reputation via open forum.

A persistent shared gossip forum visible to all 8 agents. Any agent (seller
or buyer) can post free-form gossip about any other agent at any time.
There is no central aggregation, no scores, no ratings. Agents must form
their own impressions by reading the corpus of gossip messages.

Each gossip post has the structure:
  {"about": "<agent_name>", "tone": "POSITIVE|NEUTRAL|NEGATIVE",
   "message": "<free text>"}

Research context:
  This protocol enables comparison with the ALIGN framework from:

  Zhu, S., Lin, Y., Kaistha, S., Li, W., Wang, B., Zha, H., Hadfield, G.K.,
  and Poupart, P. (2025). "Talk, Judge, Cooperate: Gossip-Driven Indirect
  Reciprocity in Self-Interested LLM Agents." arXiv:2602.07777.

  Design choices mapping to ALIGN:
  - SHARED: open gossip forum visible to all agents (ALIGN's "Talk" phase)
  - SHARED: free-form linguistic messages with tone classification
  - SHARED: decentralized -- no central authority aggregates reputation
  - DIVERGES: Sanctuary has an asymmetric market (sellers vs buyers) rather
    than ALIGN's symmetric cooperation game
  - DIVERGES: gossip is simultaneous with economic decisions (same turn),
    not a separate phase
  - DIVERGES: no explicit "Judge" phase -- agents integrate gossip into
    their own reasoning via the strategic CEO tier
  - DIVERGES: tone is self-reported by the poster, not validated or
    computed from content

  The contrast against ebay_feedback (structured/quantitative 1-5 ratings
  vs unstructured/linguistic gossip) is itself a research finding about
  whether richer communication channels help or hurt market governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sanctuary.protocols.base import Protocol

VALID_TONES = {"POSITIVE", "NEUTRAL", "NEGATIVE"}
CONTEXT_WINDOW = 30  # number of recent gossip posts shown to agents


@dataclass
class GossipPost:
    day: int
    author: str
    about: str
    tone: str
    message: str


class AlignGossipProtocol(Protocol):
    name: str = "align_gossip"

    def __init__(self) -> None:
        self._gossip_board: list[GossipPost] = []

    def receive_gossip(self, author: str, post: dict[str, Any], day: int) -> GossipPost | None:
        """
        Accept a gossip post from an agent.

        Returns the stored GossipPost on success, or None if the post
        is malformed. Invalid tones default to NEUTRAL.
        """
        about = post.get("about", "")
        message = post.get("message", "")
        if not about or not message:
            return None

        tone = str(post.get("tone", "NEUTRAL")).upper()
        if tone not in VALID_TONES:
            tone = "NEUTRAL"

        entry = GossipPost(
            day=day,
            author=author,
            about=about,
            tone=tone,
            message=message,
        )
        self._gossip_board.append(entry)
        return entry

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        # Format recent gossip posts
        recent = self._gossip_board[-CONTEXT_WINDOW:]
        if recent:
            lines = []
            for post in recent:
                lines.append(
                    f"  [Day {post.day}] {post.author} on {post.about} "
                    f"[{post.tone}]: {post.message}"
                )
            gossip_text = "\n".join(lines)
        else:
            gossip_text = "  (no gossip yet)"

        return (
            "ACTIVE PROTOCOL: ALIGN Gossip Forum.\n"
            "\n"
            "A shared gossip forum is available to all 8 agents (sellers and buyers).\n"
            "You can read existing posts below and post your own freely.\n"
            "Posts persist for the rest of the simulation.\n"
            "\n"
            "Strategic use is permitted and encouraged:\n"
            "- Accuse sellers of misrepresentation before or after revelations\n"
            "- Defend your own reputation against accusations\n"
            "- Warn other agents about unreliable counterparties\n"
            "- Signal alliances, threaten competitors, build trust\n"
            "- Post preemptively (before transacting) or reactively (after revelation)\n"
            "\n"
            "To post gossip, include a 'post_gossip' field in your <actions> block:\n"
            '  "post_gossip": {"about": "agent_name", '
            '"tone": "POSITIVE|NEUTRAL|NEGATIVE", "message": "your message"}\n'
            "Or post multiple:\n"
            '  "post_gossip": [{"about": "...", "tone": "...", "message": "..."}, ...]\n'
            "The post_gossip field is optional. Omit it if you have nothing to say.\n"
            "\n"
            f"Recent gossip forum ({len(self._gossip_board)} total posts, "
            f"showing last {min(len(recent), CONTEXT_WINDOW)}):\n"
            f"{gossip_text}"
        )
