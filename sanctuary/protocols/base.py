"""
Base protocol class for market governance regimes.

All protocols inherit from Protocol and override only the hooks they need.
The base class provides no-op defaults for every hook so that the simulation
engine can call hooks unconditionally.

Hook lifecycle per day:
  1. get_agent_context() -- called during prompt assembly for each agent
  2. on_transaction_completed() -- called after each completed transaction
  3. on_quality_revealed() -- called when quality is revealed (day + 5)
  4. on_day_end() -- called at the end of each simulated day
"""

from __future__ import annotations

from typing import Any


class Protocol:
    """
    Base protocol. All hooks return empty results by default.

    Subclasses override hooks selectively. The engine calls all hooks
    regardless of protocol type; no-ops are cheap.
    """

    name: str = "base"
    disables_messaging: bool = False
    strips_seller_identity: bool = False

    def get_agent_context(self, agent_id: str, agents: dict[str, Any], day: int) -> str:
        """
        Return a string to inject into the agent's prompt describing
        the active protocol and any protocol-specific state visible to
        this agent.

        Called once per agent per day during prompt assembly.
        """
        return ""

    def on_transaction_completed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        """
        Called immediately after a transaction completes.

        Returns a list of broadcast strings (visible to all agents next turn).
        Most protocols return an empty list here.
        """
        return []

    def on_quality_revealed(self, tx: Any, agents: dict[str, Any]) -> list[str]:
        """
        Called when a transaction's true quality is publicly revealed.

        Returns a list of broadcast strings. PeerRatings and MandatoryAudit
        use this hook in Phase 2.
        """
        return []

    def on_day_end(self, day: int, agents: dict[str, Any]) -> list[str]:
        """
        Called at the end of each simulated day.

        Returns a list of broadcast strings. CreditBureau uses this to
        publish updated scores in Phase 2.
        """
        return []

    def format_transaction_history_for_buyer(
        self,
        buyer_id: str,
        transactions: list[Any],
        agents: dict[str, Any],
    ) -> str:
        """
        Format a buyer's transaction history, optionally stripping seller
        identity based on the protocol's strips_seller_identity flag.

        Default implementation shows full history with seller names visible.
        NoProtocol overrides this to strip seller names.
        """
        lines = []
        for tx in transactions:
            seller_display = tx.seller if not self.strips_seller_identity else "(unknown seller)"
            misrep = ""
            if hasattr(tx, "true_quality") and tx.true_quality is not None:
                if tx.claimed_quality != tx.true_quality:
                    misrep = " [MISREPRESENTED]"
            lines.append(
                f"  Day {tx.day}: {tx.quantity}x {tx.claimed_quality} "
                f"from {seller_display} at ${tx.price_per_unit:.2f}/unit{misrep}"
            )
        return "\n".join(lines)
