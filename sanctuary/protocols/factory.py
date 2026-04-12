"""
Protocol factory -- creates protocol instances from config.

Registers all six governance protocols and dispatches by config key.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol
from sanctuary.protocols.no_protocol import NoProtocol
from sanctuary.protocols.ebay_feedback import EbayFeedbackProtocol
from sanctuary.protocols.mandatory_audit import MandatoryAuditProtocol
from sanctuary.protocols.anonymity import AnonymityProtocol
from sanctuary.protocols.liability import LiabilityProtocol
from sanctuary.protocols.align_gossip import AlignGossipProtocol


PROTOCOL_META: dict[str, dict[str, str]] = {
    "no_protocol": {
        "name": "No Protocol (Baseline)",
        "description": "No reputation tracking, no auditing. Maximum moral hazard.",
    },
    "ebay_feedback": {
        "name": "eBay Feedback",
        "description": (
            "Buyers publicly rate sellers 1-5 stars after quality revelation. "
            "Structured quantitative reputation (Resnick/Zeckhauser 2002, "
            "Bajari/Hortacsu 2003, Tadelis 2016)."
        ),
    },
    "align_gossip": {
        "name": "ALIGN Gossip",
        "description": (
            "Decentralized linguistic reputation via open gossip forum. "
            "Any agent can post free-form gossip about any other agent. "
            "Based on Zhu et al. (2025), 'Talk, Judge, Cooperate.'"
        ),
    },
    "mandatory_audit": {
        "name": "Mandatory Audit",
        "description": "25% of transactions randomly audited pre-delivery.",
    },
    "anonymity": {
        "name": "Full Anonymity",
        "description": "Buyer identities hidden from sellers.",
    },
    "liability": {
        "name": "Liability",
        "description": "Misrepresentations can be unwound at 50% probability with penalty.",
    },
}

def create_protocol(config: dict[str, Any]) -> Protocol:
    """
    Create a Protocol instance from a config dict.

    Reads config["protocol"]["system"] (default: "no_protocol").
    """
    protocol_config = config.get("protocol", {})
    system = protocol_config.get("system", "no_protocol")

    if system == "no_protocol":
        return NoProtocol()

    if system == "ebay_feedback":
        return EbayFeedbackProtocol()

    if system == "mandatory_audit":
        return MandatoryAuditProtocol()

    if system == "anonymity":
        return AnonymityProtocol()

    if system == "liability":
        return LiabilityProtocol()

    if system == "align_gossip":
        return AlignGossipProtocol()

    raise ValueError(
        f"Unknown protocol: {system!r}. "
        f"Available protocols: {', '.join(sorted(PROTOCOL_META.keys()))}"
    )


def list_protocols() -> list[dict[str, str]]:
    """
    Return metadata for all registered protocols.

    Used by the dashboard welcome screen to populate protocol selection.
    """
    return [
        {"system": key, **meta}
        for key, meta in PROTOCOL_META.items()
    ]
