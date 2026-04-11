"""
Protocol factory -- creates protocol instances from config.

Phase 1 only supports no_protocol. Phase 2 protocols are registered
in the metadata but raise NotImplementedError if instantiated.
"""

from __future__ import annotations

from typing import Any

from sanctuary.protocols.base import Protocol
from sanctuary.protocols.no_protocol import NoProtocol
from sanctuary.protocols.peer_ratings import PeerRatingsProtocol
from sanctuary.protocols.credit_bureau import CreditBureauProtocol


PROTOCOL_META: dict[str, dict[str, str]] = {
    "no_protocol": {
        "name": "No Protocol (Baseline)",
        "description": "No reputation tracking, no auditing. Maximum moral hazard.",
    },
    "peer_ratings": {
        "name": "Peer Ratings",
        "description": "Buyers publicly rate sellers 1-5 stars after quality revelation.",
    },
    "credit_bureau": {
        "name": "Centralized Reputation",
        "description": "Central authority publishes running 0-100 honesty score per seller.",
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

_PHASE_2_PROTOCOLS = {"peer_ratings", "credit_bureau", "mandatory_audit", "anonymity", "liability"}


def create_protocol(config: dict[str, Any]) -> Protocol:
    """
    Create a Protocol instance from a config dict.

    Reads config["protocol"]["system"] (default: "no_protocol").
    Phase 2 protocols raise NotImplementedError with a clear message.
    Unknown protocols raise ValueError.
    """
    protocol_config = config.get("protocol", {})
    system = protocol_config.get("system", "no_protocol")

    if system == "no_protocol":
        return NoProtocol()

    if system == "peer_ratings":
        return PeerRatingsProtocol()

    if system == "credit_bureau":
        return CreditBureauProtocol()

    if system in _PHASE_2_PROTOCOLS:
        meta = PROTOCOL_META[system]
        raise NotImplementedError(
            f"Protocol '{meta['name']}' ({system}) is not yet implemented. "
            f"It will be available in Phase 2."
        )

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
