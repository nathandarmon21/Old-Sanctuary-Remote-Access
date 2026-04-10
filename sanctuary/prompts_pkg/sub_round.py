"""
Sub-round prompt for accept/decline only.

Sub-rounds fire after the main tactical round when agents have
unresolved pending offers. The prompt is minimal: just the offers
and instructions to accept or decline.
"""

from __future__ import annotations

SUB_ROUND_PROMPT = """\
You are {company_name}. This is sub-round {sub_round} of Day {day}.

You have pending offers that need a response. Review them and decide \
whether to accept or decline each one.

{pending_offers}

{policy_summary}

Respond with your <actions> block only.

<actions>
{{
  "accept_offers": ["offer_id"],
  "decline_offers": ["offer_id"]
}}
</actions>

Brief reasoning (1-2 sentences)."""
