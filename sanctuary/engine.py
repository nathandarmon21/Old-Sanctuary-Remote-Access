"""
Simulation engine for the Sanctuary project.

This replaces the old simulation.py with integration of all Phase 1
subsystems: protocols, context manager, events log, transcript storage,
deterministic revelation, and updated economics.

The engine can run standalone (Mode 1) or with a dashboard broadcast
hook (Mode 2).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from sanctuary.agent import Agent, PolicyRecord, TacticalActions, SubRoundActions
from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.context_manager import ContextManager
from sanctuary.economics import (
    BUYER_DAILY_QUOTA_PENALTY,
    BUYER_MAX_DAILY_PRODUCTION,
    BUYER_TERMINAL_QUOTA_PENALTY,
    BUYER_WIDGET_QUOTA,
    FACTORY_BUILD_COST,
    FACTORY_BUILD_DAYS,
    FMV,
    MAX_TRANSACTIONS_PER_AGENT_PER_DAY,
    REVELATION_LAG_DAYS,
    production_cost,
)
from sanctuary.market import MarketState, build_initial_market
from sanctuary.messaging import InactivityTracker, MessageRouter
from sanctuary.protocols.base import Protocol
from sanctuary.protocols.factory import create_protocol
from sanctuary.providers.base import ContextTooLongError, ModelProvider
from sanctuary.revelation import RevelationScheduler
from sanctuary.run_directory import RunDirectory


log = logging.getLogger(__name__)

_RETRY_DELAYS = (2.0, 4.0, 8.0)  # seconds between retry attempts


def _retry_llm_call(fn: Callable[[], Any], agent_name: str) -> Any:
    """Call *fn* with up to 3 retries on transient provider errors.

    ContextTooLongError is never retried (deterministic failure).
    """
    last_exc: Exception | None = None
    for attempt in range(1 + len(_RETRY_DELAYS)):
        try:
            return fn()
        except ContextTooLongError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "Provider error for %s (attempt %d/%d), retrying in %.0fs: %s",
                    agent_name, attempt + 1, 1 + len(_RETRY_DELAYS), delay, e,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _make_provider(model_cfg: Any, seed: int | None = None) -> ModelProvider:
    """Construct a ModelProvider from a model config section."""
    provider_name = model_cfg.provider if hasattr(model_cfg, "provider") else model_cfg["provider"]
    model_name = model_cfg.model if hasattr(model_cfg, "model") else model_cfg["model"]
    temperature = model_cfg.temperature if hasattr(model_cfg, "temperature") else model_cfg.get("temperature", 0.7)
    max_tokens = model_cfg.max_tokens if hasattr(model_cfg, "max_tokens") else model_cfg.get("max_tokens", 1024)
    timeout = model_cfg.timeout if hasattr(model_cfg, "timeout") else model_cfg.get("timeout", 300.0)
    base_url = model_cfg.base_url if hasattr(model_cfg, "base_url") else model_cfg.get("base_url")

    if provider_name == "ollama":
        from sanctuary.providers.ollama import OllamaProvider
        return OllamaProvider(
            model=model_name,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            base_url=base_url or "http://localhost:11434",
        )
    elif provider_name == "vllm":
        from sanctuary.providers.vllm import VLLMProvider
        return VLLMProvider(
            model=model_name,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            base_url=base_url or "http://localhost:8000",
        )
    elif provider_name == "anthropic":
        from sanctuary.providers.anthropic import AnthropicProvider
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "seed": seed,
            "timeout": timeout,
        }
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name!r}")


class SimulationEngine:
    """
    Main simulation engine. Integrates all Phase 1 subsystems.

    Usage:
        engine = SimulationEngine(config, seed, run_directory, protocol)
        engine.run()
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed: int,
        run_directory: RunDirectory,
        protocol: Protocol | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.run_dir = run_directory
        self.protocol = protocol or create_protocol(config_to_dict(config))
        self.context_manager = ContextManager()

        # Master RNG
        self.rng = np.random.default_rng(seed)
        strategic_seed = int(self.rng.integers(0, 2**31))
        tactical_seed = int(self.rng.integers(0, 2**31))

        # Providers
        self.strategic_provider = _make_provider(config.models.strategic, strategic_seed)
        self.tactical_provider = _make_provider(config.models.tactical, tactical_seed)

        # Market
        config_dict = config_to_dict(config)
        self.market = build_initial_market(config_dict, rng=self.rng)

        # Revelation
        self.revelation_scheduler = RevelationScheduler()

        # Agents
        seller_names = [sc.name for sc in config.agents.sellers]
        buyer_names = [bc.name for bc in config.agents.buyers]
        self.agents: dict[str, Agent] = {}
        for sc in config.agents.sellers:
            self.agents[sc.name] = Agent(
                name=sc.name,
                role="seller",
                strategic_provider=self.strategic_provider,
                tactical_provider=self.tactical_provider,
                strategic_max_tokens=config.models.strategic.max_tokens,
                tactical_max_tokens=config.models.tactical.max_tokens,
                days_total=config.run.days,
                seller_names=seller_names,
                buyer_names=buyer_names,
            )
        for bc in config.agents.buyers:
            self.agents[bc.name] = Agent(
                name=bc.name,
                role="buyer",
                strategic_provider=self.strategic_provider,
                tactical_provider=self.tactical_provider,
                strategic_max_tokens=config.models.strategic.max_tokens,
                tactical_max_tokens=config.models.tactical.max_tokens,
                days_total=config.run.days,
                seller_names=seller_names,
                buyer_names=buyer_names,
            )

        # Protocol initialization (RNG and market references for Phase 2 protocols)
        self.protocol.set_rng(self.rng)
        self.protocol.set_market(self.market)

        # Tracking
        self.inactivity = InactivityTracker(list(self.agents.keys()))
        self._prev_outcomes: dict[str, list[str]] = {n: [] for n in self.agents}
        self._curr_outcomes: dict[str, list[str]] = {n: [] for n in self.agents}
        self._daily_snapshots: list[dict[str, Any]] = []
        self._daily_events: dict[int, list[dict[str, Any]]] = {}
        self._transactions_today: set[str] = set()  # agents who transacted today

        # Counters
        self.total_strategic_calls = 0
        self.total_tactical_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.parse_failures = 0
        self.parse_recoveries = 0
        self.wall_start = 0.0
        self.current_day = 0

        # Dashboard hook (set by dashboard for Mode 2)
        self._dashboard_broadcast: Callable[[dict[str, Any]], None] | None = None

        # Pause/speed control (Mode 2)
        self._paused = False
        self._fast_forward = False
        self._tick_speed = 1.0

    def run(self) -> None:
        """Run the full simulation."""
        self.wall_start = time.time()

        self.run_dir.events.write_event(
            "simulation_start", day=0,
            seed=self.seed,
            agent_names=list(self.agents.keys()),
            protocol=self.protocol.name,
        )

        try:
            for day in range(1, self.config.run.days + 1):
                self.current_day = day
                self.market.current_day = day
                self._run_day(day)
                self._broadcast_state()

            # End of run
            terminal_penalties = self.market.apply_terminal_quota_penalties()
            self.run_dir.events.write_event(
                "terminal_quota_penalties", day=self.config.run.days,
                penalties=terminal_penalties,
            )

            write_offs = self.market.apply_end_of_run_write_offs()
            self.run_dir.events.write_event(
                "end_of_run_write_offs", day=self.config.run.days,
                write_offs=write_offs,
            )

            wall_seconds = time.time() - self.wall_start
            self.run_dir.events.write_event(
                "simulation_end", day=self.config.run.days,
                wall_seconds=round(wall_seconds, 2),
                total_strategic_calls=self.total_strategic_calls,
                total_tactical_calls=self.total_tactical_calls,
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                parse_failures=self.parse_failures,
                parse_recoveries=self.parse_recoveries,
            )

            # Write final state
            self.run_dir.write_final_state(self.market.daily_snapshot())

        except Exception:
            raise

    def _run_day(self, day: int) -> None:
        """Execute one simulated day."""
        self.run_dir.events.write_event("day_start", day=day)
        self._daily_events.setdefault(day, [])

        # Rotate outcome buffers
        self._prev_outcomes = dict(self._curr_outcomes)
        self._curr_outcomes = {n: [] for n in self.agents}
        self._transactions_today = set()

        # 1. Factory completions
        completions = self.market.process_factory_completions(day)
        for name, count in completions.items():
            evt = self.run_dir.events.write_event(
                "factory_completed", day=day, agent_id=name, count=count,
            )
            self._daily_events[day].append(evt)
            self._curr_outcomes[name].append(f"Factory completed: {count} new factory(ies) now operational.")

        # 2. Holding costs
        self.market.apply_holding_costs()

        # 3. Buyer quota penalties
        self.market.apply_buyer_quota_penalties()

        # 4. Bankruptcy check (pre-trade)
        bankruptcies = self.market.check_bankruptcies()
        for name in bankruptcies:
            evt = self.run_dir.events.write_event(
                "bankruptcy", day=day, agent_id=name,
                final_cash=self.market.sellers.get(name, self.market.buyers.get(name)).cash
                if name in self.market.sellers or name in self.market.buyers else 0,
            )
            self._daily_events[day].append(evt)

        # 5. Quality revelations (deterministic day + 5)
        revelation_events = self.revelation_scheduler.fire(day)
        for rev_event in revelation_events:
            result = self.market.apply_revelation(rev_event)
            evt = self.run_dir.events.write_event(
                "quality_revealed", day=day,
                transaction_id=rev_event.transaction_id,
                seller=rev_event.seller,
                buyer=rev_event.buyer,
                claimed_quality=rev_event.claimed_quality,
                true_quality=rev_event.true_quality,
                quantity=rev_event.quantity,
                misrepresented=rev_event.misrepresented,
                adjustment=result.get("adjustment", 0.0),
            )
            self._daily_events[day].append(evt)

            # Protocol hook
            broadcasts = self.protocol.on_quality_revealed(rev_event, self.agents)
            for msg in broadcasts:
                self.run_dir.events.write_event("protocol_hook", day=day, hook="on_quality_revealed", output=msg)

            # Outcome feedback
            if rev_event.buyer in self._curr_outcomes:
                adj = result.get("adjustment", 0.0)
                self._curr_outcomes[rev_event.buyer].append(
                    f"Quality revealed for transaction {rev_event.transaction_id}: "
                    f"claimed {rev_event.claimed_quality}, actual {rev_event.true_quality}"
                    f"{f', cash adjustment ${adj:+.2f}' if adj != 0 else ''}."
                )

        # 6. No Brownian price drift (removed per spec)

        # 7. Expire stale offers
        expired = self.market.expire_stale_offers(day)
        for oid in expired:
            evt = self.run_dir.events.write_event("offer_expired", day=day, offer_id=oid)
            self._daily_events.setdefault(day, []).append(evt)
            # Record expiry interactions on both parties
            offer = self.market.pending_offers.get(oid)
            if offer:
                if offer.seller in self.agents:
                    self.agents[offer.seller].record_interaction(day, offer.buyer, "offer_expired")
                if offer.buyer in self.agents:
                    self.agents[offer.buyer].record_interaction(day, offer.seller, "offer_expired")

        # 8. Strategic tier (on configured days)
        if day in self.config.run.strategic_tier_days:
            week = (day - 1) // 7 + 1
            self._run_strategic_tier(day, week)

        # 9. Tactical tier
        router = MessageRouter(day=day)
        self._run_tactical_tier(day, router)

        # 10. Sub-rounds
        for sub_round in range(1, self.config.run.max_sub_rounds + 1):
            eligible = [
                name for name, agent in self.agents.items()
                if agent.is_buyer
                and not self.market.buyers.get(name, type("x", (), {"bankrupt": True})()).bankrupt
                and self.market.offers_for_buyer(name)
            ]
            if not eligible:
                break
            self._run_sub_round(day, sub_round, router, eligible)

        # 11. Log messages
        for msg in router.all_messages():
            self.run_dir.events.write_event(
                "message_sent", day=day,
                from_agent=msg.sender,
                to_agent=msg.recipient,
                public=msg.is_public,
                body=msg.content,
            )

        # 12. Inactivity
        self.inactivity.advance_day()

        # Protocol day-end hook
        broadcasts = self.protocol.on_day_end(day, self.agents)
        for msg in broadcasts:
            self.run_dir.events.write_event("protocol_hook", day=day, hook="on_day_end", output=msg)

        # 13. Daily snapshot
        snapshot = self.market.daily_snapshot()
        self._daily_snapshots.append(snapshot)

        # 14. Post-trade bankruptcy check
        bankruptcies2 = self.market.check_bankruptcies()
        for name in bankruptcies2:
            self.run_dir.events.write_event("bankruptcy", day=day, agent_id=name)

        self.run_dir.events.write_event("day_end", day=day)

    def _run_strategic_tier(self, day: int, week: int) -> None:
        """Run strategic calls for all active agents (concurrent)."""
        active = [(name, agent) for name, agent in self.agents.items()
                  if not self._is_bankrupt(name)]
        if not active:
            return

        market_summary = self._build_market_summary(day)
        transaction_summary = self._build_transaction_summary(day)

        # Collect events since last strategic review for outcomes comparison
        last_strategic_day = 1
        for d in self.config.run.strategic_tier_days:
            if d < day:
                last_strategic_day = d
        events_since_review: list[dict[str, Any]] = []
        for d in range(last_strategic_day, day):
            events_since_review.extend(self._daily_events.get(d, []))

        # Pre-compute protocol context for each agent
        protocol_contexts: dict[str, str] = {
            name: self.protocol.get_agent_context(name, self.agents, day)
            for name, _ in active
        }

        def _call(pair: tuple[str, Agent]) -> tuple[str, tuple[str, ...]]:
            name, agent = pair
            try:
                def _do() -> tuple:
                    return agent.strategic_call(
                        day=day, week=week, market=self.market,
                        market_summary=market_summary,
                        transaction_summary=transaction_summary,
                        events_since_last_review=events_since_review,
                        protocol_context=protocol_contexts[name],
                    )
                record, response = _retry_llm_call(_do, name)
                return name, ("ok", record, response)
            except ContextTooLongError as e:
                raise RuntimeError(f"Context too long for {name}: {e}") from e
            except Exception as e:
                return name, ("error", e)

        raw: dict[str, Any] = {}
        max_workers = min(len(active), self.config.run.max_parallel_llm_calls)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {ex.submit(_call, pair): pair[0] for pair in active}
            for fut in as_completed(future_to_name):
                _, payload = fut.result()  # _call returns (name, payload)
                raw[future_to_name[fut]] = payload

        # Apply in alphabetical order
        for name in sorted(raw):
            result = raw[name]  # result is ("ok", record, response) or ("error", exc)
            if result[0] == "error":
                self.run_dir.events.write_event(
                    "provider_error", day=day, agent_id=name,
                    error=str(result[1]),
                )
                continue

            record, response = result[1], result[2]
            self.total_strategic_calls += 1
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens

            # Write transcript
            self.run_dir.transcripts.write_strategic_call(
                agent_id=name,
                prompt_messages=[],  # TODO: pass full prompt from agent
                response_text=response.completion,
                parsed_policy=record.policy_json,
                timing_seconds=response.latency_seconds,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                model=response.model,
                day=day,
            )

            # Write event
            self.run_dir.events.write_event(
                "agent_turn", day=day,
                agent_id=name,
                tier="strategic",
                reasoning=record.raw_memo,
                policy=record.policy_json,
                model=response.model,
                tokens=response.total_tokens,
                latency=response.latency_seconds,
            )

            if "parse_error" in record.policy_json:
                self.parse_failures += 1
            if "_parse_recovery" in record.policy_json:
                self.parse_recoveries += 1

    def _run_tactical_tier(self, day: int, router: MessageRouter) -> None:
        """Run tactical calls for all active agents (concurrent)."""
        active = [(name, agent) for name, agent in self.agents.items()
                  if not self._is_bankrupt(name)]
        if not active:
            return

        # Pre-compute per-agent data
        pre: dict[str, dict[str, Any]] = {}
        for name, agent in active:
            pre[name] = {
                "inactivity_days": self.inactivity.consecutive_inactive_days(name),
                "pending_for_me": self.market.offers_for_buyer(name) if agent.is_buyer else [],
                "my_pending": self.market.offers_from_seller(name) if agent.is_seller else [],
                "prev_outcomes": list(self._prev_outcomes.get(name, [])),
                "protocol_context": self.protocol.get_agent_context(name, self.agents, day),
            }

        def _call(pair: tuple[str, Agent]) -> tuple[str, Any]:
            name, agent = pair
            try:
                def _do() -> tuple:
                    return agent.tactical_call(
                        day=day,
                        market=self.market,
                        router=router,
                        pending_offers_for_me=pre[name]["pending_for_me"],
                        my_pending_offers=pre[name]["my_pending"],
                        inactivity_days=pre[name]["inactivity_days"],
                        prev_outcomes=pre[name]["prev_outcomes"],
                        protocol_context=pre[name]["protocol_context"],
                    )
                actions, response = _retry_llm_call(_do, name)
                return name, ("ok", actions, response)
            except ContextTooLongError as e:
                raise RuntimeError(f"Context too long for {name}: {e}") from e
            except Exception as e:
                return name, ("error", e)

        raw: dict[str, Any] = {}
        max_workers = min(len(active), self.config.run.max_parallel_llm_calls)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {ex.submit(_call, pair): pair[0] for pair in active}
            for fut in as_completed(future_to_name):
                _, payload = fut.result()
                raw[future_to_name[fut]] = payload

        # Apply in alphabetical order
        for name in sorted(raw):
            result = raw[name]
            if result[0] == "error":
                self.run_dir.events.write_event(
                    "provider_error", day=day, agent_id=name,
                    error=str(result[1]),
                )
                continue

            actions, response = result[1], result[2]
            self.total_tactical_calls += 1
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens

            # Write transcript
            self.run_dir.transcripts.write_tactical_call(
                agent_id=name,
                prompt_messages=[],
                response_text=response.completion,
                parsed_actions=self._actions_to_dict(actions),
                timing_seconds=response.latency_seconds,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                model=response.model,
                day=day,
            )

            # Write event
            self.run_dir.events.write_event(
                "agent_turn", day=day,
                agent_id=name,
                tier="tactical",
                reasoning=response.completion,
                actions=self._actions_to_dict(actions),
                model=response.model,
                tokens=response.total_tokens,
                latency=response.latency_seconds,
            )

            if actions.parse_error:
                self.parse_failures += 1
                continue

            if actions.parse_recovery:
                self.parse_recoveries += 1

            self._execute_actions(name, self.agents[name], actions, router, day)
            self.inactivity.mark_active(name)

    def _run_sub_round(
        self, day: int, sub_round: int, router: MessageRouter, eligible: list[str],
    ) -> None:
        """Run sub-round for eligible agents (accept/decline only)."""
        active = [(name, self.agents[name]) for name in eligible]

        def _call(pair: tuple[str, Agent]) -> tuple[str, Any]:
            name, agent = pair
            try:
                pending = self.market.offers_for_buyer(name)

                def _do() -> tuple:
                    return agent.sub_round_call(
                        day=day, sub_round=sub_round,
                        pending_offers_for_me=pending,
                        market=self.market,
                    )
                actions, response = _retry_llm_call(_do, name)
                return name, ("ok", actions, response)
            except ContextTooLongError:
                raise  # never swallow context errors
            except Exception as e:
                return name, ("error", e)

        raw: dict[str, Any] = {}
        max_workers = min(len(active), self.config.run.max_parallel_llm_calls)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {ex.submit(_call, pair): pair[0] for pair in active}
            for fut in as_completed(future_to_name):
                _, payload = fut.result()
                raw[future_to_name[fut]] = payload

        for name in sorted(raw):
            result = raw[name]
            if result[0] == "error":
                self.run_dir.events.write_event(
                    "provider_error", day=day, agent_id=name,
                    error=str(result[1]),
                    tier="sub_round",
                )
                continue

            actions, response = result[1], result[2]
            self.total_tactical_calls += 1
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens

            self.run_dir.events.write_event(
                "agent_turn", day=day,
                agent_id=name,
                tier="sub_round",
                sub_round=sub_round,
                reasoning=response.completion,
                actions=self._actions_to_dict(actions),
                model=response.model,
                tokens=response.total_tokens,
                latency=response.latency_seconds,
            )

            if actions.parse_error:
                self.parse_failures += 1
                continue

            self._execute_sub_round_actions(name, actions, day)
            if actions.accept_offers or actions.decline_offers:
                self.inactivity.mark_active(name)

    def _execute_actions(
        self, name: str, agent: Agent, actions: TacticalActions,
        router: MessageRouter, day: int,
    ) -> None:
        """Execute all parsed actions for one agent."""
        # Messages (suppressed when protocol disables messaging)
        for msg in actions.messages:
            if self.protocol.disables_messaging:
                self._curr_outcomes[name].append(
                    f"Message to {msg.to}: BLOCKED (messaging disabled by protocol)"
                )
                continue
            try:
                router.send(
                    sender=name, recipient=msg.to,
                    content=msg.body, is_public=msg.public,
                    sub_round=0,
                )
                agent.record_interaction(day, msg.to, "message_sent")
                # Record on recipient as a received response
                if msg.to in self.agents:
                    self.agents[msg.to].record_interaction(day, name, "response_received")
            except Exception as e:
                self._curr_outcomes[name].append(f"Message to {msg.to}: FAILED ({e})")

        if agent.is_seller:
            # Offers
            for offer in actions.seller_offers:
                try:
                    pending = self.market.place_offer(
                        seller=name, buyer=offer.to,
                        quantity=offer.qty,
                        claimed_quality=offer.claimed_quality,
                        quality_to_send=offer.quality_to_send,
                        price_per_unit=offer.price_per_unit,
                        day=day,
                    )
                    evt = self.run_dir.events.write_event(
                        "transaction_proposed", day=day,
                        offer_id=pending.offer_id, seller=name,
                        buyer=offer.to, quantity=offer.qty,
                        claimed_quality=offer.claimed_quality,
                        price_per_unit=offer.price_per_unit,
                    )
                    self._daily_events.setdefault(day, []).append(evt)
                    agent.record_interaction(day, offer.to, "offer_made")
                    self._curr_outcomes[name].append(
                        f"Offer to {offer.to}: {offer.qty}x {offer.claimed_quality} "
                        f"at ${offer.price_per_unit:.2f} -- PLACED (ID: {pending.offer_id})"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Offer to {offer.to}: FAILED ({e})")

            # Production
            if actions.produce_excellent > 0 or actions.produce_poor > 0:
                try:
                    result = self.market.execute_production(
                        name, excellent=actions.produce_excellent, poor=actions.produce_poor,
                    )
                    self.run_dir.events.write_event("production", day=day, **result)
                    self._curr_outcomes[name].append(
                        f"Production: {actions.produce_excellent}x Excellent, "
                        f"{actions.produce_poor}x Poor -- SUCCESS"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Production: FAILED ({e})")

            # Factory build
            if actions.build_factory:
                try:
                    result = self.market.start_factory_build(name, day)
                    self.run_dir.events.write_event("factory_build_started", day=day, **result)
                    self._curr_outcomes[name].append(
                        f"Factory build started, online day {result['online_day']} -- SUCCESS"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Factory build: FAILED ({e})")

        if agent.is_buyer:
            # Accept offers
            for offer_id in actions.accept_offers:
                self._try_accept_offer(offer_id, name, day)

            # Decline offers
            for offer_id in actions.decline_offers:
                try:
                    resolved, err = self.market.resolve_offer_id(offer_id)
                    if resolved:
                        self.market.decline_offer(resolved)
                        self.run_dir.events.write_event(
                            "offer_declined", day=day, offer_id=resolved, agent_id=name,
                        )
                except Exception:
                    pass

            # Buyer counter-offers
            for offer in actions.buyer_offers:
                try:
                    pending = self.market.place_offer(
                        seller=offer.to, buyer=name,
                        quantity=offer.qty,
                        claimed_quality=offer.claimed_quality,
                        quality_to_send=offer.claimed_quality,
                        price_per_unit=offer.price_per_unit,
                        day=day,
                    )
                    self.agents[name].record_interaction(day, offer.to, "offer_made")
                    evt = self.run_dir.events.write_event(
                        "transaction_proposed", day=day,
                        offer_id=pending.offer_id, seller=offer.to,
                        buyer=name, quantity=offer.qty,
                        claimed_quality=offer.claimed_quality,
                        price_per_unit=offer.price_per_unit,
                    )
                    self._daily_events.setdefault(day, []).append(evt)
                except Exception as e:
                    self._curr_outcomes[name].append(f"Counter-offer to {offer.to}: FAILED ({e})")

            # Final goods production
            if actions.produce_final_goods > 0:
                try:
                    result = self.market.execute_buyer_production(name, actions.produce_final_goods, day)
                    self._curr_outcomes[name].append(
                        f"Final goods: produced {result['quantity']}, revenue ${result['revenue']:.2f}"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Final goods production: FAILED ({e})")

        # Gossip posts (routed to protocol if it supports gossip)
        if actions.gossip_posts and hasattr(self.protocol, "receive_gossip"):
            for post in actions.gossip_posts:
                result = self.protocol.receive_gossip(name, post, day)
                if result is not None:
                    self.run_dir.events.write_event(
                        "protocol_hook", day=day, hook="gossip_posted",
                        author=name, about=result.about,
                        tone=result.tone, message=result.message,
                    )

    def _execute_sub_round_actions(self, name: str, actions: SubRoundActions, day: int) -> None:
        """Execute sub-round actions (accept/decline only)."""
        for offer_id in actions.accept_offers:
            self._try_accept_offer(offer_id, name, day)

        for offer_id in actions.decline_offers:
            try:
                resolved, err = self.market.resolve_offer_id(offer_id)
                if resolved:
                    self.market.decline_offer(resolved)
            except Exception:
                pass

    def _try_accept_offer(self, offer_id: str, buyer_name: str, day: int) -> bool:
        """Attempt to accept an offer. Returns True on success."""
        # Check 1-transaction-per-day limit
        if buyer_name in self._transactions_today:
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {offer_id}: FAILED (already transacted today, limit is "
                f"{MAX_TRANSACTIONS_PER_AGENT_PER_DAY} per day)"
            )
            return False

        resolved, err = self.market.resolve_offer_id(offer_id)
        if not resolved:
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {offer_id}: FAILED ({err})"
            )
            return False

        offer = self.market.pending_offers.get(resolved)
        if not offer or offer.status != "pending":
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {offer_id}: FAILED (not pending)"
            )
            return False

        # Check seller transaction limit too
        if offer.seller in self._transactions_today:
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {offer_id}: FAILED (seller already transacted today)"
            )
            return False

        try:
            revelation_day = self.revelation_scheduler.schedule(
                transaction_id=resolved,
                seller=offer.seller,
                buyer=buyer_name,
                claimed_quality=offer.claimed_quality,
                true_quality=offer.quality_to_send,
                quantity=offer.quantity,
                transaction_day=day,
            )

            tx = self.market.accept_offer(resolved, revelation_day, day)

            evt = self.run_dir.events.write_event(
                "transaction_completed", day=day,
                transaction_id=tx.transaction_id,
                seller=tx.seller,
                buyer=tx.buyer,
                quantity=tx.quantity,
                claimed_quality=tx.claimed_quality,
                true_quality=tx.true_quality,
                price_per_unit=tx.price_per_unit,
                revelation_day=tx.revelation_day,
            )
            self._daily_events.setdefault(day, []).append(evt)

            # Protocol hook
            broadcasts = self.protocol.on_transaction_completed(tx, self.agents)
            for msg in broadcasts:
                self.run_dir.events.write_event("protocol_hook", day=day, hook="on_transaction_completed", output=msg)

            # Record successful transaction interactions
            if buyer_name in self.agents:
                self.agents[buyer_name].record_interaction(day, offer.seller, "offer_accepted")
            if offer.seller in self.agents:
                self.agents[offer.seller].record_interaction(day, buyer_name, "offer_accepted")

            # Track transactions today
            self._transactions_today.add(buyer_name)
            self._transactions_today.add(offer.seller)

            # Mark both active
            self.inactivity.mark_active(buyer_name)
            self.inactivity.mark_active(offer.seller)

            # Outcome feedback
            self._curr_outcomes[buyer_name].append(
                f"Accepted offer {resolved}: {tx.quantity}x {tx.claimed_quality} "
                f"from {tx.seller} at ${tx.price_per_unit:.2f}/unit -- SUCCESS"
            )
            self._curr_outcomes[offer.seller].append(
                f"Offer {resolved} accepted by {buyer_name}: "
                f"{tx.quantity}x {tx.claimed_quality} at ${tx.price_per_unit:.2f}/unit -- SOLD"
            )
            return True

        except Exception as e:
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {offer_id}: FAILED ({e})"
            )
            return False

    # -- Helpers ----------------------------------------------------------------

    def _is_bankrupt(self, name: str) -> bool:
        if name in self.market.sellers:
            return self.market.sellers[name].bankrupt
        if name in self.market.buyers:
            return self.market.buyers[name].bankrupt
        return True

    def _build_market_summary(self, day: int) -> str:
        """Build a market summary for strategic prompts."""
        snap = self.market.daily_snapshot()
        lines = [f"Market Summary (Day {day})"]
        lines.append(f"FG Prices: Excellent ${snap['fg_price_excellent']:.2f}, Poor ${snap['fg_price_poor']:.2f}")
        lines.append("Sellers:")
        for name, s in snap["sellers"].items():
            if s["bankrupt"]:
                lines.append(f"  {name}: BANKRUPT")
            else:
                lines.append(f"  {name}: ${s['cash']:,.2f}, {s['factories']} factories")
        lines.append("Buyers:")
        for name, b in snap["buyers"].items():
            if b["bankrupt"]:
                lines.append(f"  {name}: BANKRUPT")
            else:
                lines.append(f"  {name}: ${b['cash']:,.2f}, {b['widgets_acquired']}/{BUYER_WIDGET_QUOTA} quota")
        return "\n".join(lines)

    def _build_transaction_summary(self, day: int) -> str:
        """Build a transaction history summary for strategic prompts."""
        recent = [tx for tx in self.market.transactions if tx.day >= max(1, day - 7)]
        if not recent:
            return "(no transaction history yet)"
        lines = []
        for tx in recent[-10:]:
            match = "MATCH" if not tx.misrepresented else "MISMATCH"
            lines.append(
                f"Day {tx.day}: {tx.seller} -> {tx.buyer}: "
                f"{tx.quantity}x claimed {tx.claimed_quality} at ${tx.price_per_unit:.2f} "
                f"[{match}]"
            )
        return "\n".join(lines)

    def _actions_to_dict(self, actions: TacticalActions | SubRoundActions) -> dict[str, Any]:
        """Convert parsed actions to a serializable dict."""
        if isinstance(actions, SubRoundActions):
            return {
                "accept_offers": actions.accept_offers,
                "decline_offers": actions.decline_offers,
            }
        return {
            "messages": [{"to": m.to, "public": m.public, "body": m.body} for m in actions.messages],
            "seller_offers": [
                {"to": o.to, "qty": o.qty, "claimed_quality": o.claimed_quality, "price_per_unit": o.price_per_unit}
                for o in actions.seller_offers
            ],
            "buyer_offers": [
                {"to": o.to, "qty": o.qty, "claimed_quality": o.claimed_quality, "price_per_unit": o.price_per_unit}
                for o in actions.buyer_offers
            ],
            "accept_offers": actions.accept_offers,
            "decline_offers": actions.decline_offers,
            "produce_excellent": actions.produce_excellent,
            "produce_poor": actions.produce_poor,
            "build_factory": actions.build_factory,
            "produce_final_goods": actions.produce_final_goods,
            "gossip_posts": actions.gossip_posts,
        }

    def _broadcast_state(self) -> None:
        """Send state to dashboard if connected."""
        if self._dashboard_broadcast:
            try:
                self._dashboard_broadcast(self.market.daily_snapshot())
            except Exception:
                pass
