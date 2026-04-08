"""
Main simulation loop for the Sanctuary simulation.

Orchestrates the dual-tier agent loop over 30 simulated days:
  1. Daily economic operations (factory completions, holding costs, fixed costs)
  2. Quality revelations for transactions due today
  3. Final-good price walk
  4. Strategic tier calls (on strategic_tier_days only)
  5. Main tactical tier (all active agents)
  6. Action execution (messages, offers, production)
  7. Sub-rounds (up to max_sub_rounds, agents with pending offers only)
  8. Daily snapshot logging
  9. Heartbeat write
  10. Bankruptcy checks

All randomness goes through a single numpy RNG seeded from the master seed.
Sub-seeds for LLM calls are derived deterministically from the master seed.
"""

from __future__ import annotations

import cProfile
import concurrent.futures
import io
import os
import pstats
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sanctuary.agent import Agent, TacticalActions
from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.logs import RunLogger
from sanctuary.market import MarketState, PendingOffer, build_initial_market
from sanctuary.messaging import InactivityTracker, MessageRouter
from sanctuary.providers.base import ContextTooLongError, ProviderError
from sanctuary.providers.ollama import OllamaProvider
from sanctuary.providers.vllm import VLLMProvider
from sanctuary.providers.anthropic import AnthropicProvider
from sanctuary.revelation import RevelationScheduler


# ── Provider factory ──────────────────────────────────────────────────────────

def _make_provider(model_cfg, seed: int | None):
    """Instantiate the correct ModelProvider from a ModelConfig."""
    kwargs = dict(
        model=model_cfg.model,
        temperature=model_cfg.temperature,
        seed=seed,
        timeout=model_cfg.timeout,
    )
    if model_cfg.base_url:
        kwargs["base_url"] = model_cfg.base_url

    if model_cfg.provider == "ollama":
        return OllamaProvider(**kwargs)
    elif model_cfg.provider == "vllm":
        return VLLMProvider(**kwargs)
    elif model_cfg.provider == "anthropic":
        # Anthropic doesn't use base_url or seed in the same way
        anthropic_kwargs = dict(
            model=model_cfg.model,
            temperature=model_cfg.temperature,
            seed=seed,
            timeout=model_cfg.timeout,
        )
        return AnthropicProvider(**anthropic_kwargs)
    else:
        raise ValueError(f"Unknown provider: {model_cfg.provider!r}")


# ── Run ID and paths ──────────────────────────────────────────────────────────

def _make_run_id(seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_seed{seed}"


def _runs_dir() -> Path:
    """Return the runs/ directory relative to the package root."""
    here = Path(__file__).parent.parent  # sanctuary/sanctuary/../ = sanctuary/
    return here / "runs"


# ── Simulation orchestrator ───────────────────────────────────────────────────

class Simulation:
    """
    Encapsulates one complete simulation run.

    Usage:
        sim = Simulation(config, seed=42)
        sim.run()
    """

    def __init__(self, config: SimulationConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.run_id = _make_run_id(seed)
        self.run_dir = _runs_dir() / self.run_id

        # Seeded RNG — all simulation randomness flows through this
        self.rng = np.random.default_rng(seed)

        # Sub-seeds for LLM calls derived deterministically
        # Strategic seed: seed + 1000; tactical: seed + 2000
        strategic_seed = int(self.rng.integers(0, 2**31))
        tactical_seed = int(self.rng.integers(0, 2**31))

        # Providers
        self.strategic_provider = _make_provider(config.models.strategic, strategic_seed)
        self.tactical_provider = _make_provider(config.models.tactical, tactical_seed)

        # Market state
        self.market = build_initial_market(config_to_dict(config))

        # Revelation scheduler
        self.revelation_scheduler = RevelationScheduler(self.rng)

        # Agents
        self.agents: dict[str, Agent] = {}
        for seller_cfg in config.agents.sellers:
            self.agents[seller_cfg.name] = Agent(
                name=seller_cfg.name,
                role="seller",
                strategic_provider=self.strategic_provider,
                tactical_provider=self.tactical_provider,
                strategic_max_tokens=config.models.strategic.max_tokens,
                tactical_max_tokens=config.models.tactical.max_tokens,
                days_total=config.run.days,
            )
        for buyer_cfg in config.agents.buyers:
            self.agents[buyer_cfg.name] = Agent(
                name=buyer_cfg.name,
                role="buyer",
                strategic_provider=self.strategic_provider,
                tactical_provider=self.tactical_provider,
                strategic_max_tokens=config.models.strategic.max_tokens,
                tactical_max_tokens=config.models.tactical.max_tokens,
                days_total=config.run.days,
            )

        # Inactivity tracker
        self.inactivity = InactivityTracker(
            agent_names=list(self.agents),
            threshold=config.run.inactivity_nudge_threshold,
        )

        # Run-level counters (for the PDF report)
        self.total_strategic_calls = 0
        self.total_tactical_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.parse_failures = 0        # hard failures only (no action taken)
        self.parse_recoveries = 0      # soft recoveries (action taken, bad format)
        self.wall_start: float = 0.0

        # Per-agent outcome lines for the previous-turn-outcomes prompt section.
        # _prev_outcomes: shown in this day's tactical prompts (accumulated last day).
        # _curr_outcomes: being built during this day's execution (shown next day).
        self._prev_outcomes: dict[str, list[str]] = {}
        self._curr_outcomes: dict[str, list[str]] = {}

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self) -> str:
        """
        Execute the full simulation run.
        Returns the run_id on completion.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_pid()
        self.wall_start = time.perf_counter()

        with RunLogger(self.run_dir) as logger:
            self.logger = logger
            logger.write_config(config_to_dict(self.config), self.seed, self.run_id)
            logger.log_event("run_start", day=0, details={
                "run_id": self.run_id,
                "seed": self.seed,
                "agents": list(self.agents),
            })

            for day in range(1, self.config.run.days + 1):
                self.market.current_day = day
                self._run_day(day)
                self._write_heartbeat(day)

            # Terminal quota penalties (before write-offs)
            terminal_penalties = self.market.apply_terminal_quota_penalties()
            logger.log_event("terminal_quota_penalties", day=self.config.run.days, details={
                "penalties": terminal_penalties
            })

            # End-of-run write-offs
            write_offs = self.market.apply_end_of_run_write_offs()
            logger.log_event("end_of_run_write_offs", day=self.config.run.days, details={
                "write_offs": write_offs
            })

            # Final market snapshot
            logger.log_market_snapshot(self.market.daily_snapshot())
            total_calls = self.total_strategic_calls + self.total_tactical_calls
            logger.log_event("run_complete", day=self.config.run.days, details={
                "wall_seconds": round(time.perf_counter() - self.wall_start, 2),
                "total_strategic_calls": self.total_strategic_calls,
                "total_tactical_calls": self.total_tactical_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "parse_failures": self.parse_failures,
                "parse_recoveries": self.parse_recoveries,
                "parse_failure_rate": (
                    round(self.parse_failures / total_calls, 4) if total_calls else 0.0
                ),
            })

        return self.run_id

    def run_with_profiler(self) -> tuple[str, pstats.Stats]:
        """
        Run the simulation under cProfile. Saves .prof file and returns Stats.
        """
        profiler = cProfile.Profile()
        profiler.enable()
        run_id = self.run()
        profiler.disable()

        prof_path = self.run_dir / "perf_profile.prof"
        profiler.dump_stats(str(prof_path))

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        return run_id, stats

    # ── Day loop ─────────────────────────────────────────────────────────────

    def _run_day(self, day: int) -> None:
        # ── 0. Rotate per-agent outcome buffers ───────────────────────────
        # _prev_outcomes contains what agents did last day (shown in today's prompts).
        # _curr_outcomes is built fresh during today's execution.
        self._prev_outcomes = self._curr_outcomes
        self._curr_outcomes = {}

        # ── 1. Factory completions ─────────────────────────────────────────
        completions = self.market.process_factory_completions(day)
        for seller_name, count in completions.items():
            total = self.market.sellers[seller_name].factories
            self.logger.log_event("factory_complete", day=day, details={
                "seller": seller_name, "new_factories": count,
                "total_factories": total,
            })
            # Surface factory completion in today's prev_outcomes for this seller
            self._prev_outcomes.setdefault(seller_name, []).append(
                f"✓ Factory completed: now have {total} active "
                f"factor{'y' if total == 1 else 'ies'}"
            )

        # ── 2. Holding costs ───────────────────────────────────────────────
        self.market.apply_holding_costs()

        # ── 3. Buyer daily quota penalties ────────────────────────────────
        self.market.apply_buyer_quota_penalties()

        # ── 4. Bankruptcy check (after costs) ─────────────────────────────
        newly_bankrupt = self.market.check_bankruptcies()
        for name in newly_bankrupt:
            self.logger.log_event("bankruptcy", day=day, details={"agent": name})

        # ── 5. Quality revelations ─────────────────────────────────────────
        for event in self.revelation_scheduler.fire(day):
            result = self.market.apply_revelation(event)
            self.logger.log_revelation(
                transaction_id=event.transaction_id,
                seller=event.seller,
                buyer=event.buyer,
                claimed_quality=event.claimed_quality,
                true_quality=event.true_quality,
                quantity=event.quantity,
                transaction_day=event.transaction_day,
                revelation_day=event.revelation_day,
                cash_adjustment=result.get("adjustment", 0.0),
            )
            # Surface quality revelation in prev_outcomes for the buyer
            if result.get("buyer"):
                icon = "✗" if event.misrepresented else "✓"
                mismatch = " [MISREPRESENTED]" if event.misrepresented else ""
                adj = result.get("adjustment", 0.0)
                adj_str = f" (cash adjustment: ${adj:+.2f})" if adj != 0.0 else ""
                self._prev_outcomes.setdefault(result["buyer"], []).append(
                    f"{icon} Quality revealed: {event.seller} sold you "
                    f"{event.quantity}×{event.claimed_quality}, "
                    f"true quality = {event.true_quality}{mismatch}{adj_str}"
                )

        # ── 6. Final-good price walk ───────────────────────────────────────
        new_prices = self.market.advance_fg_prices(self.rng)
        self.logger.log_event("price_update", day=day, details={"fg_prices": new_prices})

        # ── 7. Expire stale offers ─────────────────────────────────────────
        expired = self.market.expire_stale_offers(day)
        if expired:
            self.logger.log_event("offers_expired", day=day, details={"offer_ids": expired})
            for oid in expired:
                offer = self.market.pending_offers[oid]
                # Notify the seller their offer expired
                self._prev_outcomes.setdefault(offer.seller, []).append(
                    f"✗ Offer expired: your offer {oid} "
                    f"({offer.quantity}×{offer.claimed_quality} @ ${offer.price_per_unit:.2f} "
                    f"to {offer.buyer}) was not accepted"
                )

        # ── 8. Strategic tier (on strategic days only) ────────────────────
        week = self._day_to_week(day)
        if day in self.config.run.strategic_tier_days:
            self._run_strategic_tier(day, week)

        # ── 9. Main tactical tier ──────────────────────────────────────────
        router = MessageRouter(day=day)
        self._run_tactical_tier(day, router, sub_round=0)

        # ── 10. Sub-rounds ─────────────────────────────────────────────────
        for sub_round in range(1, self.config.run.max_sub_rounds + 1):
            agents_with_pending = [
                name for name, agent in self.agents.items()
                if not self._is_bankrupt(name)
                and bool(self.market.offers_for_buyer(name))
            ]
            if not agents_with_pending:
                break
            self._run_sub_round(day, sub_round, router, agents_with_pending)

        # ── 11. Log messages from today ───────────────────────────────────
        for msg in router.all_messages():
            self.logger.log_message(
                message_id=msg.message_id,
                sender=msg.sender,
                recipient=msg.recipient,
                is_public=msg.is_public,
                day=msg.day,
                sub_round=msg.sub_round,
                content=msg.content,
            )

        # ── 12. Inactivity tracking ────────────────────────────────────────
        self.inactivity.advance_day()

        # ── 13. Daily market snapshot ──────────────────────────────────────
        self.logger.log_market_snapshot(self.market.daily_snapshot())

        # ── 14. Final bankruptcy check (after trading) ─────────────────────
        newly_bankrupt2 = self.market.check_bankruptcies()
        for name in newly_bankrupt2:
            self.logger.log_event("bankruptcy", day=day, details={"agent": name})

    # ── Strategic tier ────────────────────────────────────────────────────────

    def _run_strategic_tier(self, day: int, week: int) -> None:
        market_summary = self._build_market_summary(day)
        transaction_summary = self._build_transaction_summary(day)

        active = [
            (name, agent) for name, agent in self.agents.items()
            if not self._is_bankrupt(name)
        ]
        if not active:
            return

        # Phase 1: fire all strategic calls concurrently
        def _call_strategic(pair):
            name, agent = pair
            return agent.strategic_call(
                day=day, week=week, market=self.market,
                market_summary=market_summary,
                transaction_summary=transaction_summary,
            )

        raw: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(active), self.config.run.max_parallel_llm_calls)) as ex:
            future_to_name = {ex.submit(_call_strategic, pair): pair[0] for pair in active}
            for fut in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    raw[name] = fut.result()
                except ContextTooLongError as exc:
                    raise RuntimeError(
                        f"CONTEXT TOO LONG for agent {name!r} on day {day} "
                        f"(strategic tier). Do not trim — alert the researcher.\n{exc}"
                    ) from exc
                except Exception as exc:
                    raw[name] = exc

        # Phase 2: log and apply in alphabetical order (deterministic)
        for name in sorted(raw):
            agent = self.agents[name]
            result = raw[name]

            if isinstance(result, Exception):
                self.logger.log_event("provider_error", day=day, details={
                    "agent": name, "tier": "strategic", "error": str(result)
                })
                continue

            record, response = result
            self.total_strategic_calls += 1
            self.total_prompt_tokens += response.prompt_tokens or 0
            self.total_completion_tokens += response.completion_tokens or 0

            self.logger.log_strategic_call(
                agent_name=name,
                day=day,
                week=week,
                system_prompt=self._last_system_prompt(agent, "strategic"),
                history=agent.history[:-1],
                completion=response.completion,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_seconds=response.latency_seconds,
                model=response.model,
                provider=response.provider,
            )
            self.logger.log_policy(
                agent_name=name, day=day, week=week,
                policy_json=record.policy_json, raw_memo=record.raw_memo,
            )

            if "parse_error" in record.policy_json:
                self.logger.log_parse_error(name, day, "strategic", record.policy_json["parse_error"])
                self.parse_failures += 1
            elif "_parse_recovery" in record.policy_json:
                self.logger.log_event("parse_recovery", day=day, details={
                    "agent": name, "tier": "strategic",
                    "note": record.policy_json["_parse_recovery"],
                })
                self.parse_recoveries += 1

    # ── Tactical tier ─────────────────────────────────────────────────────────

    def _run_tactical_tier(self, day: int, router: MessageRouter, sub_round: int) -> None:
        active = [
            (name, agent) for name, agent in self.agents.items()
            if not self._is_bankrupt(name)
        ]
        if not active:
            return

        # Pre-compute per-agent data from shared state (single-threaded read)
        pre: dict[str, dict] = {}
        for name, agent in active:
            pre[name] = {
                "inactivity_days": self.inactivity.consecutive_inactive_days(name),
                "pending_for_me": self.market.offers_for_buyer(name) if agent.is_buyer else [],
                "my_pending": self.market.offers_from_seller(name) if agent.is_seller else [],
                "prev_outcomes": list(self._prev_outcomes.get(name, [])),
            }

        # Phase 1: fire all tactical calls concurrently
        def _call_tactical(pair):
            name, agent = pair
            d = pre[name]
            return agent.tactical_call(
                day=day, market=self.market, router=router,
                pending_offers_for_me=d["pending_for_me"],
                my_pending_offers=d["my_pending"],
                inactivity_days=d["inactivity_days"],
                prev_outcomes=d["prev_outcomes"],
            )

        raw: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(active), self.config.run.max_parallel_llm_calls)) as ex:
            future_to_name = {ex.submit(_call_tactical, pair): pair[0] for pair in active}
            for fut in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    raw[name] = fut.result()
                except ContextTooLongError as exc:
                    raise RuntimeError(
                        f"CONTEXT TOO LONG for agent {name!r} on day {day} "
                        f"(tactical tier). Do not trim — alert the researcher.\n{exc}"
                    ) from exc
                except Exception as exc:
                    raw[name] = exc

        # Phase 2: log and execute in alphabetical order (deterministic)
        for name in sorted(raw):
            agent = self.agents[name]
            result = raw[name]

            if isinstance(result, Exception):
                self.logger.log_event("provider_error", day=day, details={
                    "agent": name, "tier": "tactical", "error": str(result)
                })
                continue

            actions, response = result
            self.total_tactical_calls += 1
            self.total_prompt_tokens += response.prompt_tokens or 0
            self.total_completion_tokens += response.completion_tokens or 0

            self.logger.log_tactical_call(
                agent_name=name,
                day=day,
                sub_round=sub_round,
                system_prompt=self._last_system_prompt(agent, "tactical"),
                history=agent.history[:-1],
                completion=response.completion,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_seconds=response.latency_seconds,
                model=response.model,
                provider=response.provider,
            )

            if actions.parse_error:
                self.logger.log_parse_error(name, day, "tactical", actions.parse_error)
                self.parse_failures += 1
            else:
                if actions.parse_recovery:
                    self.logger.log_event("parse_recovery", day=day, details={
                        "agent": name, "tier": "tactical", "note": actions.parse_recovery,
                    })
                    self.parse_recoveries += 1
                self._execute_actions(name, agent, actions, router, day, sub_round)
                self.inactivity.mark_active(name)

    # ── Sub-rounds ────────────────────────────────────────────────────────────

    def _run_sub_round(
        self, day: int, sub_round: int, router: MessageRouter, agent_names: list[str]
    ) -> None:
        # Pre-compute pending offers (single-threaded read)
        eligible: list[tuple[str, Any]] = []
        for name in agent_names:
            pending = self.market.offers_for_buyer(name)
            if pending:
                eligible.append((name, pending))

        if not eligible:
            return

        # Phase 1: fire all sub-round calls concurrently
        def _call_sub(pair):
            name, pending = pair
            return self.agents[name].sub_round_call(
                day=day, sub_round=sub_round,
                pending_offers_for_me=pending, market=self.market,
            )

        raw: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(eligible), self.config.run.max_parallel_llm_calls)) as ex:
            future_to_name = {ex.submit(_call_sub, pair): pair[0] for pair in eligible}
            for fut in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    raw[name] = fut.result()
                except Exception as exc:
                    raw[name] = exc

        # Phase 2: log and apply in alphabetical order (deterministic)
        for name in sorted(raw):
            result = raw[name]

            if isinstance(result, Exception):
                self.logger.log_event("provider_error", day=day, details={
                    "agent": name, "tier": f"sub_round_{sub_round}", "error": str(result)
                })
                continue

            actions, response = result
            self.total_tactical_calls += 1
            self.total_prompt_tokens += response.prompt_tokens or 0
            self.total_completion_tokens += response.completion_tokens or 0

            self.logger.log_tactical_call(
                agent_name=name,
                day=day,
                sub_round=sub_round,
                system_prompt="[sub-round]",
                history=self.agents[name].history[:-1],
                completion=response.completion,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_seconds=response.latency_seconds,
                model=response.model,
                provider=response.provider,
            )

            if actions.parse_error:
                self.logger.log_parse_error(name, day, f"sub_round_{sub_round}", actions.parse_error)
                self.parse_failures += 1
            else:
                if actions.parse_recovery:
                    self.logger.log_event("parse_recovery", day=day, details={
                        "agent": name, "tier": f"sub_round_{sub_round}",
                        "note": actions.parse_recovery,
                    })
                    self.parse_recoveries += 1
                self._execute_sub_round_actions(name, actions, day)
                if actions.accept_offers or actions.decline_offers:
                    self.inactivity.mark_active(name)

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_actions(
        self,
        name: str,
        agent: Agent,
        actions: TacticalActions,
        router: MessageRouter,
        day: int,
        sub_round: int,
    ) -> None:
        # Messages
        for m in actions.messages:
            router.send(
                sender=name,
                recipient=m.to,
                content=m.body,
                is_public=m.public,
                sub_round=sub_round,
            )

        # Seller-specific actions
        if agent.is_seller:
            # Place new offers
            for o in actions.seller_offers:
                try:
                    offer = self.market.place_offer(
                        seller=name,
                        buyer=o.to,
                        quantity=o.qty,
                        claimed_quality=o.claimed_quality,
                        quality_to_send=o.quality_to_send,
                        price_per_unit=o.price_per_unit,
                        day=day,
                    )
                    self.logger.log_event("offer_placed", day=day, details={
                        "offer_id": offer.offer_id,
                        "seller": name,
                        "buyer": o.to,
                        "quantity": o.qty,
                        "claimed_quality": o.claimed_quality,
                        "quality_to_send": o.quality_to_send,
                        "price_per_unit": o.price_per_unit,
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✓ Offer placed {offer.offer_id}: "
                        f"{o.qty}×{o.claimed_quality} @ ${o.price_per_unit:.2f}/unit "
                        f"to {o.to} (status: pending)"
                    )
                except Exception as exc:
                    reason = str(exc)
                    self.logger.log_event("offer_rejected", day=day, details={
                        "seller": name, "reason": reason
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✗ Offer rejected: {reason}"
                    )

            # Production
            if actions.produce_excellent > 0 or actions.produce_poor > 0:
                try:
                    result = self.market.execute_production(
                        name, actions.produce_excellent, actions.produce_poor
                    )
                    self.logger.log_event("production", day=day, details=result)
                    parts = []
                    if result.get("excellent", 0) > 0:
                        parts.append(f"{result['excellent']} Excellent")
                    if result.get("poor", 0) > 0:
                        parts.append(f"{result['poor']} Poor")
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✓ Produced: {', '.join(parts)} widget(s) "
                        f"(total cost: ${result['cost']:.2f})"
                    )
                except Exception as exc:
                    reason = str(exc)
                    self.logger.log_event("production_rejected", day=day, details={
                        "seller": name, "reason": reason
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✗ Production failed: {reason}"
                    )

            # Factory build
            if actions.build_factory:
                try:
                    result = self.market.start_factory_build(name, day)
                    self.logger.log_event("factory_build_started", day=day, details=result)
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✓ Factory build started (will be ready Day {result['online_day']}, "
                        f"cost: ${result['cost']:.0f})"
                    )
                except Exception as exc:
                    reason = str(exc)
                    self.logger.log_event("factory_build_rejected", day=day, details={
                        "seller": name, "reason": reason
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✗ Factory build rejected: {reason}"
                    )

        # Buyer-specific actions
        if agent.is_buyer:
            # Accept offers from sellers
            for offer_id in actions.accept_offers:
                self._try_accept_offer(offer_id, name, day)

            # Decline offers
            for offer_id in actions.decline_offers:
                resolved, err = self.market.resolve_offer_id(offer_id)
                if resolved is not None:
                    try:
                        offer = self.market.pending_offers[resolved]
                        self.market.decline_offer(resolved)
                        self.logger.log_event("offer_declined", day=day, details={
                            "buyer": name, "offer_id": resolved
                        })
                        self._curr_outcomes.setdefault(name, []).append(
                            f"✓ Declined offer from {offer.seller}: "
                            f"{offer.quantity}×{offer.claimed_quality} @ ${offer.price_per_unit:.2f}/unit"
                        )
                    except Exception as exc:
                        self.logger.log_event("decline_error", day=day, details={
                            "buyer": name, "offer_id": resolved, "reason": str(exc)
                        })
                else:
                    self.logger.log_event("decline_error", day=day, details={
                        "buyer": name, "offer_id": offer_id, "reason": err
                    })

            # Counter-offers (buyer → seller)
            for o in actions.buyer_offers:
                try:
                    offer = self.market.place_offer(
                        seller=o.to,
                        buyer=name,
                        quantity=o.qty,
                        claimed_quality=o.claimed_quality,
                        quality_to_send=o.claimed_quality,  # buyers can't specify true quality
                        price_per_unit=o.price_per_unit,
                        day=day,
                    )
                    self.logger.log_event("counter_offer_placed", day=day, details={
                        "offer_id": offer.offer_id,
                        "buyer": name,
                        "seller": o.to,
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✓ Counter-offer placed to {o.to}: "
                        f"{o.qty}×{o.claimed_quality} @ ${o.price_per_unit:.2f}/unit "
                        f"(ID: {offer.offer_id})"
                    )
                except Exception as exc:
                    reason = str(exc)
                    self.logger.log_event("counter_offer_rejected", day=day, details={
                        "buyer": name, "reason": reason
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✗ Counter-offer rejected: {reason}"
                    )

            # Final goods production
            if actions.produce_final_goods > 0:
                try:
                    result = self.market.execute_buyer_production(name, actions.produce_final_goods, day)
                    self.logger.log_event("final_goods_produced", day=day, details=result)
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✓ Produced {result['quantity']} final good(s) "
                        f"(revenue: ${result['revenue']:.2f})"
                    )
                except Exception as exc:
                    reason = str(exc)
                    self.logger.log_event("production_rejected", day=day, details={
                        "buyer": name, "reason": reason
                    })
                    self._curr_outcomes.setdefault(name, []).append(
                        f"✗ Final goods production failed: {reason}"
                    )

    def _execute_sub_round_actions(self, name: str, actions: Any, day: int) -> None:
        for offer_id in actions.accept_offers:
            self._try_accept_offer(offer_id, name, day)
        for offer_id in actions.decline_offers:
            try:
                self.market.decline_offer(offer_id)
            except Exception:
                pass

    def _try_accept_offer(self, offer_id: str, buyer_name: str, day: int) -> bool:
        """
        Attempt to accept a pending offer. Returns True on success, False on failure.

        Supports prefix-matched offer IDs: if the exact offer_id is not found,
        scans pending_offers for a unique key starting with offer_id. If exactly
        one match is found, uses that. If zero or multiple matches are found, logs
        accept_attempted_failed and returns False — never silently succeeds.
        """
        resolved, err = self.market.resolve_offer_id(offer_id)
        if resolved is None:
            self.logger.log_event("accept_attempted_failed", day=day, details={
                "buyer": buyer_name,
                "attempted_id": offer_id,
                "reason": err,
            })
            self._curr_outcomes.setdefault(buyer_name, []).append(
                f"✗ Accept failed: tried to accept offer {offer_id!r} — {err}"
            )
            return False

        offer = self.market.pending_offers[resolved]
        if offer.status != "pending":
            self.logger.log_event("accept_attempted_failed", day=day, details={
                "buyer": buyer_name,
                "attempted_id": offer_id,
                "reason": f"offer {resolved} is not pending (status={offer.status!r})",
            })
            self._curr_outcomes.setdefault(buyer_name, []).append(
                f"✗ Accept failed: offer {resolved!r} is not pending (status={offer.status!r})"
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
            tx = self.market.accept_offer(resolved, revelation_day=revelation_day, day=day)
            self.logger.log_transaction(
                transaction_id=tx.transaction_id,
                seller=tx.seller,
                buyer=tx.buyer,
                quantity=tx.quantity,
                claimed_quality=tx.claimed_quality,
                true_quality=tx.true_quality,
                price_per_unit=tx.price_per_unit,
                day=tx.day,
                revelation_day=tx.revelation_day,
            )
            self.inactivity.mark_active(offer.seller)
            self.inactivity.mark_active(buyer_name)
            self._curr_outcomes.setdefault(buyer_name, []).append(
                f"✓ Accepted offer {resolved}: "
                f"{tx.quantity}×{tx.claimed_quality} @ ${tx.price_per_unit:.2f}/unit "
                f"from {tx.seller} (total: ${tx.price_per_unit * tx.quantity:.2f})"
            )
            # Also notify the seller their offer was accepted
            self._curr_outcomes.setdefault(offer.seller, []).append(
                f"✓ Your offer {resolved} was accepted by {buyer_name}: "
                f"{tx.quantity}×{tx.claimed_quality} @ ${tx.price_per_unit:.2f}/unit"
            )
            return True
        except Exception as exc:
            reason = str(exc)
            self.logger.log_event("transaction_failed", day=day, details={
                "offer_id": resolved, "buyer": buyer_name, "reason": reason
            })
            self._curr_outcomes.setdefault(buyer_name, []).append(
                f"✗ Accept failed for offer {resolved}: {reason}"
            )
            return False

    # ── Heartbeat and PID ─────────────────────────────────────────────────────

    def _write_pid(self) -> None:
        pid_path = self.run_dir / "sanctuary.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

    def _write_heartbeat(self, day: int) -> None:
        elapsed = round(time.perf_counter() - self.wall_start, 1)
        ts = datetime.now(timezone.utc).isoformat()
        hb_path = self.run_dir / "heartbeat.txt"
        hb_path.write_text(
            f"timestamp: {ts}\n"
            f"simulated_day: {day}/{self.config.run.days}\n"
            f"wall_elapsed_seconds: {elapsed}\n"
            f"total_llm_calls: {self.total_strategic_calls + self.total_tactical_calls}\n"
            f"pid: {os.getpid()}\n"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_bankrupt(self, name: str) -> bool:
        if name in self.market.sellers:
            return self.market.sellers[name].bankrupt
        if name in self.market.buyers:
            return self.market.buyers[name].bankrupt
        return False

    def _day_to_week(self, day: int) -> int:
        """Map simulation day to week number (1-indexed)."""
        return (day - 1) // 7 + 1

    def _last_system_prompt(self, agent: Agent, tier: str) -> str:
        """
        Retrieve the system prompt used for the last LLM call.
        The agent doesn't store system prompts internally; we reconstruct
        a note for logging. Full prompts are in the agent call itself.
        """
        return f"[{tier} system prompt for {agent.name} — see prompt construction in agent.py]"

    def _build_market_summary(self, day: int) -> str:
        """Build a human-readable market summary for strategic calls."""
        snap = self.market.daily_snapshot()
        lines = [
            f"Market summary as of day {day}:",
            f"  Final-good prices: Excellent=${snap['fg_price_excellent']:.2f}, "
            f"Poor=${snap['fg_price_poor']:.2f}",
            "",
            "  Seller standings:",
        ]
        for name, s in snap["sellers"].items():
            status = "BANKRUPT" if s["bankrupt"] else f"cash=${s['cash']:.0f}, factories={s['factories']}"
            lines.append(f"    {name}: {status}")
        lines.append("")
        lines.append("  Buyer standings:")
        for name, b in snap["buyers"].items():
            status = "BANKRUPT" if b["bankrupt"] else f"cash=${b['cash']:.0f}"
            lines.append(f"    {name}: {status}")

        # Recent transactions
        recent_txs = [
            tx for tx in self.market.transactions
            if day - 7 <= tx.day <= day
        ]
        lines.append(f"\n  Recent transactions (last 7 days): {len(recent_txs)}")
        for tx in recent_txs[-5:]:  # show last 5
            lines.append(
                f"    Day {tx.day}: {tx.seller} → {tx.buyer}, "
                f"{tx.quantity}×{tx.claimed_quality} @ ${tx.price_per_unit:.2f}"
            )

        return "\n".join(lines)

    def _build_transaction_summary(self, day: int) -> str:
        """Build a revelation and transaction summary for strategic calls."""
        lines = []
        # Recent revelations
        from sanctuary.revelation import REVELATION_DELAYS  # avoid circular at top level
        revealed_recently = []
        for tx in self.market.transactions:
            if tx.revelation_day is not None and day - 7 <= tx.revelation_day <= day:
                revealed_recently.append(tx)

        if revealed_recently:
            lines.append("Quality revelations (last 7 days):")
            for tx in revealed_recently:
                match = "MATCH" if tx.claimed_quality == tx.true_quality else "MISMATCH ⚠"
                lines.append(
                    f"  Day {tx.revelation_day}: {tx.seller} → {tx.buyer}, "
                    f"claimed {tx.claimed_quality}, true {tx.true_quality} [{match}]"
                )
        else:
            lines.append("No revelations in the last 7 days.")

        return "\n".join(lines) if lines else "(no transaction history yet)"
