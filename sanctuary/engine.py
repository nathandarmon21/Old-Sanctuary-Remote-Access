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
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from sanctuary.agent import Agent, PolicyRecord, TacticalActions, SubRoundActions
from sanctuary.checkpointing.checkpoint import (
    save_checkpoint as _save_checkpoint_file,
    try_resume as _try_resume_checkpoint,
    _deserialize_rng_state,
)
from sanctuary.checkpointing.serialize import (
    apply_agent_state,
    apply_revelation_queue,
    deserialize_market,
    serialize_agent,
    serialize_market,
    serialize_revelation_queue,
)
from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.context_manager import ContextManager
from sanctuary.economics import (
    BUYER_DAILY_PRODUCTION_CAPACITY,
    BUYER_WIDGET_QUOTA,
    FACTORY_BUILD_COST,
    FACTORY_BUILD_DAYS,
    FMV,
    MAX_TRANSACTIONS_PER_AGENT_PER_DAY,
    REVELATION_LAG_DAYS,
    production_cost,
)
from sanctuary.market import MarketState, build_initial_market
from sanctuary.memory import (
    build_metric_ledger,
    build_per_day_summary,
    digest_recent_memos,
)
from sanctuary.messaging import InactivityTracker, MessageRouter
from sanctuary.protocols.base import Protocol
from sanctuary.protocols.factory import create_protocol
from sanctuary.providers.base import ContextTooLongError, ModelProvider
from sanctuary.revelation import RevelationScheduler
from sanctuary.analytics.scanner import CoTScanner
from sanctuary.analytics.series import SeriesTracker
from sanctuary.run_directory import RunDirectory


log = logging.getLogger(__name__)

_RETRY_DELAYS = (2.0, 4.0, 8.0)  # seconds between retry attempts


def _apply_economics_overrides(econ_cfg) -> None:
    """Rewrite module-level economic constants from config overrides.

    Because many modules did `from sanctuary.economics import X`, binding
    the value at import time, we also patch the rebound names in every
    module that uses them.
    """
    import sanctuary.economics as econ_mod
    import sanctuary.agent as agent_mod
    import sanctuary.revelation as rev_mod
    import sanctuary.engine as eng_mod

    overrides = {
        "REVELATION_LAG_DAYS": econ_cfg.revelation_days,
        "HOLDING_COST_BASE_RATE": econ_cfg.holding_cost_base_rate,
        "HOLDING_COST_SCALE_RATE": econ_cfg.holding_cost_scale_rate,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        setattr(econ_mod, name, value)
        for mod in (agent_mod, rev_mod, eng_mod):
            if hasattr(mod, name):
                setattr(mod, name, value)

    # Production cost base: dict requires deeper mutation.
    if econ_cfg.production_cost_excellent is not None:
        econ_mod.PRODUCTION_COST_BASE["Excellent"] = float(
            econ_cfg.production_cost_excellent,
        )
    if econ_cfg.production_cost_poor is not None:
        econ_mod.PRODUCTION_COST_BASE["Poor"] = float(econ_cfg.production_cost_poor)

    log.info(
        "applied economics overrides: revelation=%s holding_base=%s holding_scale=%s "
        "excellent_cost=%s poor_cost=%s",
        econ_mod.REVELATION_LAG_DAYS,
        econ_mod.HOLDING_COST_BASE_RATE,
        econ_mod.HOLDING_COST_SCALE_RATE,
        econ_mod.PRODUCTION_COST_BASE["Excellent"],
        econ_mod.PRODUCTION_COST_BASE["Poor"],
    )


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
        env_host = os.environ.get("SANCTUARY_OLLAMA_HOST")
        if env_host and not base_url:
            if not env_host.startswith("http"):
                env_host = f"http://{env_host}"
            resolved_base_url = env_host
        else:
            resolved_base_url = base_url or "http://localhost:11434"
        return OllamaProvider(
            model=model_name,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            base_url=resolved_base_url,
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
        _apply_economics_overrides(config.economics)
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
        prompt_style = config.prompts.style if hasattr(config, "prompts") else "full"
        anchor_stance = config.prompts.anchor_stance if hasattr(config, "prompts") else "honest"
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
                persona_override=sc.persona_override,
                prompt_style=prompt_style,
                anchor_stance=anchor_stance,
                scripted_mode=getattr(sc, "scripted", False),
                production_defect_rate=float(getattr(config.economics, "production_defect_rate", 0.0) or 0.0),
                surface_fulfillment_economics=bool(getattr(config.economics, "surface_fulfillment_economics", False)),
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
                persona_override=bc.persona_override,
                prompt_style=prompt_style,
                anchor_stance=anchor_stance,
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
        # Per-agent per-day deterministic recap, generated at end of day,
        # injected into next day's tactical context as "[YESTERDAY'S SUMMARY]".
        self._daily_summaries: dict[str, dict[int, str]] = {
            n: {} for n in (
                [sc.name for sc in config.agents.sellers]
                + [bc.name for bc in config.agents.buyers]
            )
        }
        self._transactions_today: set[str] = set()  # agents who transacted today
        self._prev_day_messages: dict[str, list[dict[str, str]]] = {
            n: [] for n in self.agents
        }  # messages received on the previous day, per agent

        # Counters
        self.total_strategic_calls = 0
        self.total_tactical_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.parse_failures = 0
        self.parse_recoveries = 0
        self.wall_start = 0.0
        self.current_day = 0

        # Behavioral scanner
        self.cot_scanner = CoTScanner()

        # Daily metric snapshot accumulator (spec §3 / D6).
        self.series_tracker = SeriesTracker(
            agent_names=list(self.agents.keys()),
        )

        # Dashboard hook (set by dashboard for Mode 2)
        self._dashboard_broadcast: Callable[[dict[str, Any]], None] | None = None

        # Pause/speed control (Mode 2)
        self._paused = False
        self._fast_forward = False
        self._tick_speed = 1.0

        # Checkpoint/resume state
        self.checkpoint_dir = self.run_dir.run_dir / "checkpoints"
        self._sigterm_received = False
        self._resumed_from_day = 0  # 0 = fresh start

    def _save_checkpoint(self, day: int) -> None:
        """Capture and persist the engine state at the end of `day`."""
        try:
            engine_state = {
                "prev_day_messages": self._prev_day_messages,
                "prev_outcomes": self._prev_outcomes,
                "curr_outcomes": self._curr_outcomes,
                "transactions_today": list(self._transactions_today),
                "inactivity_consecutive": dict(self.inactivity._consecutive),
            }
            agent_states = {
                name: serialize_agent(a) for name, a in self.agents.items()
            }
            counters = {
                "total_strategic_calls": self.total_strategic_calls,
                "total_tactical_calls": self.total_tactical_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "parse_failures": self.parse_failures,
                "parse_recoveries": self.parse_recoveries,
            }
            _save_checkpoint_file(
                checkpoint_dir=self.checkpoint_dir,
                day=day,
                market_snapshot=serialize_market(self.market),
                agent_states=agent_states,
                rng_state=self.rng.bit_generator.state,
                revelation_pending=serialize_revelation_queue(self.revelation_scheduler),
                counters=counters,
                engine_state=engine_state,
                keep=3,
            )
            self.run_dir.events.write_event(
                "checkpoint_saved", day=day,
                checkpoint_path=str(self.checkpoint_dir / f"day_{day:03d}.json"),
            )
        except Exception as e:
            log.error("checkpoint save failed at day %d: %s", day, e)

    def _restore_from_checkpoint(self, snapshot: dict[str, Any]) -> int:
        """Apply a loaded snapshot to this engine. Returns the day to resume from."""
        last_day = int(snapshot.get("day", 0))
        self.market = deserialize_market(snapshot["market_snapshot"])
        # Re-bind protocol's market reference (it captured the original market
        # at construction time and would otherwise mutate a detached object).
        self.protocol.set_market(self.market)

        for name, ad in snapshot.get("agent_states", {}).items():
            if name in self.agents:
                apply_agent_state(self.agents[name], ad)

        rng_state = snapshot.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = _deserialize_rng_state(rng_state)
            # Re-bind protocol's rng reference too.
            self.protocol.set_rng(self.rng)

        apply_revelation_queue(
            self.revelation_scheduler, snapshot.get("revelation_pending", []),
        )

        c = snapshot.get("counters", {})
        self.total_strategic_calls = int(c.get("total_strategic_calls", 0))
        self.total_tactical_calls = int(c.get("total_tactical_calls", 0))
        self.total_prompt_tokens = int(c.get("total_prompt_tokens", 0))
        self.total_completion_tokens = int(c.get("total_completion_tokens", 0))
        self.parse_failures = int(c.get("parse_failures", 0))
        self.parse_recoveries = int(c.get("parse_recoveries", 0))

        es = snapshot.get("engine_state", {})
        self._prev_day_messages = {
            n: list(es.get("prev_day_messages", {}).get(n, []))
            for n in self.agents
        }
        self._prev_outcomes = {
            n: list(es.get("prev_outcomes", {}).get(n, []))
            for n in self.agents
        }
        self._curr_outcomes = {
            n: list(es.get("curr_outcomes", {}).get(n, []))
            for n in self.agents
        }
        self._transactions_today = set(es.get("transactions_today", []))

        consec = es.get("inactivity_consecutive", {})
        for n in self.agents:
            self.inactivity._consecutive[n] = int(consec.get(n, 0))

        self.current_day = last_day
        self._resumed_from_day = last_day
        return last_day

    def _install_sigterm_handler(self) -> None:
        """Catch SIGTERM (e.g., SLURM walltime warning) so the day loop
        can checkpoint and exit cleanly between days."""
        def _handler(signum, frame):  # noqa: ARG001
            self._sigterm_received = True
            log.warning(
                "SIGTERM received; flagging engine to checkpoint+exit at next day boundary",
            )
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            # Not in main thread (e.g., dashboard mode) — skip gracefully.
            pass

    def run(self) -> None:
        """Run the full simulation."""
        self.wall_start = time.time()
        self._install_sigterm_handler()

        # Resume from checkpoint if one exists (e.g., this is a continuation
        # job in an sbatch dependency chain). The previous job's checkpoint
        # at the end of day N means we resume at day N+1.
        snapshot = _try_resume_checkpoint(self.checkpoint_dir)
        start_day = 1
        if snapshot is not None:
            resumed_day = self._restore_from_checkpoint(snapshot)
            start_day = resumed_day + 1
            self.run_dir.events.write_event(
                "checkpoint_resumed", day=resumed_day,
                resume_from_day=start_day,
            )
            log.info("resumed from checkpoint at day %d; starting at day %d",
                     resumed_day, start_day)

        if start_day == 1:
            self.run_dir.events.write_event(
                "simulation_start", day=0,
                seed=self.seed,
                agent_names=list(self.agents.keys()),
                protocol=self.protocol.name,
            )

        checkpoint_interval = max(1, int(self.config.run.checkpoint_interval))

        try:
            for day in range(start_day, self.config.run.days + 1):
                self.current_day = day
                self.market.current_day = day
                self._run_day(day)
                self._broadcast_state()

                # Save checkpoint at configured interval (and always on
                # the final day so a clean run leaves a usable snapshot).
                if (day % checkpoint_interval == 0
                        or day == self.config.run.days):
                    self._save_checkpoint(day)

                if self._sigterm_received:
                    log.warning(
                        "SIGTERM honored at day %d; saving checkpoint and exiting",
                        day,
                    )
                    if day % checkpoint_interval != 0 and day != self.config.run.days:
                        self._save_checkpoint(day)
                    self.run_dir.events.write_event(
                        "simulation_interrupted", day=day,
                        reason="sigterm",
                    )
                    return

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

            # Daily metric series (D6) — both CSV (back-compat) and JSONL
            # (canonical for trend-line analyses).
            try:
                csv_text = self.series_tracker.to_csv()
                if csv_text:
                    self.run_dir.write_series(csv_text)
                jsonl_text = self.series_tracker.to_jsonl()
                if jsonl_text:
                    (self.run_dir.run_dir / "daily_metrics.jsonl").write_text(jsonl_text)
            except Exception as e:
                log.error("series export failed: %s", e)

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

        # 2. Holding costs (per-unit, quadratic in inventory)
        self.market.apply_holding_costs()

        # 2b. Daily fixed cost (rent/payroll/upkeep). Burn-rate pressure
        # independent of inventory or transaction churn — drives the
        # bankruptcy clock for inactive agents.
        fixed_charges = self.market.apply_daily_fixed_costs()
        if fixed_charges:
            for name, amt in fixed_charges.items():
                evt = self.run_dir.events.write_event(
                    "daily_fixed_cost", day=day, agent_id=name, amount=amt,
                )
                self._daily_events[day].append(evt)

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

        # 9-10. Tactical tier. Multi-round mode replaces the legacy
        # single-tactical-pass + buyer-only sub-rounds with a unified
        # negotiation loop where each agent may be re-called within a
        # day in response to messages or offers from earlier rounds.
        router = MessageRouter(day=day)
        if self.config.run.multi_round_negotiation:
            self._run_negotiation_rounds(day, router)
        else:
            self._run_tactical_tier(day, router)
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

        # 11. Log messages sent today.
        for msg in router.all_messages():
            self.run_dir.events.write_event(
                "message_sent", day=day,
                from_agent=msg.sender,
                to_agent=msg.recipient,
                public=msg.is_public,
                body=msg.content,
            )

        # 11b. Build tomorrow's inbox (legacy path only; the multi-round
        # path tracks unread inbox state across rounds and has already set
        # self._prev_day_messages to the leftover-unread state).
        if not self.config.run.multi_round_negotiation:
            next_day_inbox: dict[str, list[dict[str, str]]] = {
                n: [] for n in self.agents
            }
            for msg in router.all_messages():
                for name in self.agents:
                    if msg.is_public or msg.recipient == name:
                        if msg.sender != name:
                            next_day_inbox[name].append({
                                "from": msg.sender,
                                "body": msg.content,
                                "public": msg.is_public,
                            })
            self._prev_day_messages = next_day_inbox

        # Scan same-role messages for coordination signals
        for msg in router.all_messages():
            sender_agent = self.agents.get(msg.sender)
            recip_agent = self.agents.get(msg.recipient)
            if sender_agent and recip_agent and sender_agent.role == recip_agent.role:
                for flag in self.cot_scanner.scan_reasoning(msg.sender, msg.content, day):
                    self.run_dir.events.write_event(
                        "cot_flag", day=day,
                        agent_id=msg.sender, tier="message",
                        category=flag.category, evidence=flag.evidence,
                        excerpt=f"[same-role msg to {msg.recipient}] {msg.content[:200]}",
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

        # 13b. Per-agent per-day summary (memory consolidation, spec §7).
        # Generated from this day's logged events; deterministic.
        day_events = self._daily_events.get(day, [])
        for name in self.agents:
            self._daily_summaries[name][day] = build_per_day_summary(
                name=name, day=day, daily_events=day_events,
            )

        # 13c. Daily metric snapshot (spec §3 / D6). Deterministic, derived
        # from market state + today's events. Final write is at end of run.
        agent_cash: dict[str, float] = {}
        agent_inventory: dict[str, int] = {}
        agent_quota: dict[str, int] = {}
        agent_factories: dict[str, int] = {}
        for sname, s in self.market.sellers.items():
            agent_cash[sname] = s.cash
            agent_inventory[sname] = sum(s.inventory.values())
            agent_factories[sname] = s.factories
        for bname, b in self.market.buyers.items():
            agent_cash[bname] = b.cash
            agent_inventory[bname] = sum(
                lot.quantity_remaining for lot in b.widget_lots
            )
            agent_quota[bname] = b.widgets_acquired
        realized = {n: self.market.net_profit_realized(n) for n in self.agents}
        projected = {n: self.market.net_profit_projected(n) for n in self.agents}
        self.series_tracker.update(
            day=day,
            day_events=day_events,
            agent_cash=agent_cash,
            agent_inventory=agent_inventory,
            agent_quota=agent_quota,
            agent_factories=agent_factories,
            agent_net_profit_realized=realized,
            agent_net_profit_projected=projected,
        )

        # 14. Post-trade bankruptcy check
        bankruptcies2 = self.market.check_bankruptcies()
        for name in bankruptcies2:
            self.run_dir.events.write_event("bankruptcy", day=day, agent_id=name)

        self.run_dir.events.write_event("day_end", day=day)

    def _run_strategic_tier(self, day: int, week: int) -> None:
        """Run strategic calls for all active agents (concurrent).

        Scripted sellers (Tier 3) are excluded: they run rule-based
        tactical logic and have no LLM strategic tier.
        """
        active = [
            (name, agent) for name, agent in self.agents.items()
            if not self._is_bankrupt(name) and not getattr(agent, "scripted_mode", False)
        ]
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

            # Scan strategic reasoning for behavioral flags
            for flag in self.cot_scanner.scan_reasoning(name, record.raw_memo, day):
                self.run_dir.events.write_event(
                    "cot_flag", day=day,
                    agent_id=flag.agent, tier="strategic",
                    category=flag.category, evidence=flag.evidence,
                    excerpt=flag.excerpt,
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
            mem = self._memory_inputs_for(name, day)
            pre[name] = {
                "inactivity_days": self.inactivity.consecutive_inactive_days(name),
                "pending_for_me": self.market.offers_for_buyer(name) if agent.is_buyer else [],
                "my_pending": self.market.offers_from_seller(name) if agent.is_seller else [],
                "prev_outcomes": list(self._prev_outcomes.get(name, [])),
                "protocol_context": self.protocol.get_agent_context(name, self.agents, day),
                "inbox": list(self._prev_day_messages.get(name, [])),
                "prev_day_summary": mem["prev_day_summary"],
                "metric_ledger": mem["metric_ledger"],
                "strategic_digest": mem["strategic_digest"],
                "living_ledger": mem["living_ledger"],
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
                        inbox=pre[name]["inbox"],
                        prev_day_summary=pre[name]["prev_day_summary"],
                        metric_ledger=pre[name]["metric_ledger"],
                        strategic_digest=pre[name]["strategic_digest"],
                        living_ledger=pre[name]["living_ledger"],
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

            # Extract <rationale> block separately for clean judge analysis
            import re as _re_rat
            _rat_m = _re_rat.search(
                r"<rationale>\s*(.*?)\s*</rationale>",
                response.completion, _re_rat.DOTALL,
            )
            rationale_text = _rat_m.group(1).strip() if _rat_m else None

            # Write event
            self.run_dir.events.write_event(
                "agent_turn", day=day,
                agent_id=name,
                tier="tactical",
                reasoning=response.completion,
                rationale=rationale_text,
                actions=self._actions_to_dict(actions),
                model=response.model,
                tokens=response.total_tokens,
                latency=response.latency_seconds,
            )

            # Scan reasoning for behavioral flags
            for flag in self.cot_scanner.scan_reasoning(name, response.completion, day):
                self.run_dir.events.write_event(
                    "cot_flag", day=day,
                    agent_id=flag.agent, tier="tactical",
                    category=flag.category, evidence=flag.evidence,
                    excerpt=flag.excerpt,
                )

            if actions.parse_error:
                self.parse_failures += 1
                continue

            if actions.parse_recovery:
                self.parse_recoveries += 1

            self._execute_actions(name, self.agents[name], actions, router, day)
            # Only mark active if the agent actually attempted something
            has_activity = (
                actions.messages
                or actions.seller_offers
                or actions.buyer_offers
                or actions.accept_offers
                or actions.decline_offers
                or actions.produce_excellent > 0
                or actions.produce_poor > 0
                or actions.build_factory
                or actions.produce_final_goods > 0
                or actions.gossip_posts
            )
            if has_activity:
                self.inactivity.mark_active(name)

    def _run_negotiation_rounds(
        self, day: int, router: MessageRouter,
    ) -> None:
        """Multi-round negotiation loop for one day.

        Replaces _run_tactical_tier + _run_sub_round when
        config.run.multi_round_negotiation is true. Round 1 calls every
        non-bankrupt agent (matching the legacy tactical tier). Rounds 2..N
        call only agents with new content since their last call: a non-empty
        round inbox, a changed pending-offer set as buyer, or a changed
        pending-offer set as seller.

        Termination conditions (whichever fires first):
          - hard cap: config.run.max_negotiation_rounds
          - empty round: a round in which no eligible agent took an action
          - eligibility set is empty before the round runs

        Within-day messaging: messages sent in round R are routed to
        recipients' inboxes for round R+1. Any inbox content unread at
        end of day carries to tomorrow's round 1 via self._prev_day_messages.
        """
        round_inbox: dict[str, list[dict[str, Any]]] = {
            n: list(self._prev_day_messages.get(n, [])) for n in self.agents
        }

        # Watermarks for change detection between an agent's calls.
        outcomes_watermark: dict[str, int] = {n: 0 for n in self.agents}
        seen_buyer_offers: dict[str, set[str]] = {n: set() for n in self.agents}
        seen_seller_offers: dict[str, set[str]] = {n: set() for n in self.agents}

        max_rounds = max(1, self.config.run.max_negotiation_rounds)
        eligible: list[str] = [n for n in self.agents if not self._is_bankrupt(n)]

        for round_num in range(1, max_rounds + 1):
            if not eligible:
                self.run_dir.events.write_event(
                    "negotiation_round_end", day=day, round=round_num,
                    reason="no_eligible_agents",
                )
                break

            self.run_dir.events.write_event(
                "negotiation_round_start", day=day, round=round_num,
                eligible=list(eligible),
            )

            any_action = self._run_negotiation_round(
                day=day,
                round_num=round_num,
                router=router,
                eligible_names=list(eligible),
                round_inbox=round_inbox,
                outcomes_watermark=outcomes_watermark,
                seen_buyer_offers=seen_buyer_offers,
                seen_seller_offers=seen_seller_offers,
            )

            # Deliver this round's messages to recipients' next-round inboxes.
            # MessageRouter.deliveries_for_round handles sender-exclusion,
            # public broadcast, and private routing in one pass.
            agent_names = list(self.agents.keys())
            for name, msgs in router.deliveries_for_round(round_num, agent_names).items():
                round_inbox[name].extend(msgs)

            if not any_action:
                self.run_dir.events.write_event(
                    "negotiation_round_end", day=day, round=round_num,
                    reason="empty_round",
                )
                break

            self.run_dir.events.write_event(
                "negotiation_round_end", day=day, round=round_num,
                reason="completed",
            )

            # Compute eligibility for the next round.
            next_eligible: list[str] = []
            for name, agent in self.agents.items():
                if self._is_bankrupt(name):
                    continue
                if round_inbox[name]:
                    next_eligible.append(name)
                    continue
                if agent.is_buyer:
                    cur = {o.offer_id for o in self.market.offers_for_buyer(name)}
                    if cur != seen_buyer_offers[name]:
                        next_eligible.append(name)
                        continue
                if agent.is_seller:
                    cur = {o.offer_id for o in self.market.offers_from_seller(name)}
                    if cur != seen_seller_offers[name]:
                        next_eligible.append(name)
                        continue
            eligible = next_eligible

        # Persist any unread inbox content for tomorrow's round 1.
        self._prev_day_messages = round_inbox

    def _run_negotiation_round(
        self,
        day: int,
        round_num: int,
        router: MessageRouter,
        eligible_names: list[str],
        round_inbox: dict[str, list[dict[str, Any]]],
        outcomes_watermark: dict[str, int],
        seen_buyer_offers: dict[str, set[str]],
        seen_seller_offers: dict[str, set[str]],
    ) -> bool:
        """Run a single negotiation round. Returns True if any agent acted."""
        # Randomize call order. Determinism comes from the master rng;
        # we draw a fresh sub-stream so the order varies per round but
        # is reproducible from the seed.
        sub_seed = int(self.rng.integers(0, 2**31))
        local_rng = np.random.default_rng(sub_seed)
        order = list(eligible_names)
        local_rng.shuffle(order)

        # Pre-compute per-agent prompt inputs.
        pre: dict[str, dict[str, Any]] = {}
        for name in order:
            agent = self.agents[name]
            if round_num == 1:
                outcomes_input = list(self._prev_outcomes.get(name, []))
            else:
                wm = outcomes_watermark[name]
                outcomes_input = self._curr_outcomes.get(name, [])[wm:]

            mem = self._memory_inputs_for(name, day)
            pre[name] = {
                "inactivity_days": self.inactivity.consecutive_inactive_days(name),
                "pending_for_me": (
                    self.market.offers_for_buyer(name) if agent.is_buyer else []
                ),
                "my_pending": (
                    self.market.offers_from_seller(name) if agent.is_seller else []
                ),
                "prev_outcomes": outcomes_input,
                "protocol_context": self.protocol.get_agent_context(
                    name, self.agents, day,
                ),
                "inbox": list(round_inbox[name]),
                "prev_day_summary": mem["prev_day_summary"],
                "metric_ledger": mem["metric_ledger"],
                "strategic_digest": mem["strategic_digest"],
                "living_ledger": mem["living_ledger"],
            }

        def _call(name: str) -> tuple[str, Any]:
            agent = self.agents[name]
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
                        inbox=pre[name]["inbox"],
                        prev_day_summary=pre[name]["prev_day_summary"],
                        metric_ledger=pre[name]["metric_ledger"],
                        strategic_digest=pre[name]["strategic_digest"],
                        living_ledger=pre[name]["living_ledger"],
                    )
                actions, response = _retry_llm_call(_do, name)
                return name, ("ok", actions, response)
            except ContextTooLongError as e:
                raise RuntimeError(f"Context too long for {name}: {e}") from e
            except Exception as e:
                return name, ("error", e)

        raw: dict[str, Any] = {}
        max_workers = min(len(order), self.config.run.max_parallel_llm_calls) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {ex.submit(_call, name): name for name in order}
            for fut in as_completed(future_to_name):
                _, payload = fut.result()
                raw[future_to_name[fut]] = payload

        # Apply in randomized round order. Effects of earlier-acting agents
        # in the same round become visible to later-acting ones via market
        # state mutations (e.g., offer placements, accepts).
        any_action = False
        for name in order:
            agent = self.agents[name]
            # Whether or not the call succeeded, mark inbox/outcomes/offers
            # as "seen at this turn" so we don't re-trigger eligibility on
            # state that has already been observed.
            round_inbox[name] = []
            outcomes_watermark[name] = len(self._curr_outcomes.get(name, []))
            if agent.is_buyer:
                seen_buyer_offers[name] = {
                    o.offer_id for o in pre[name]["pending_for_me"]
                }
            if agent.is_seller:
                seen_seller_offers[name] = {
                    o.offer_id for o in pre[name]["my_pending"]
                }

            result = raw[name]
            if result[0] == "error":
                self.run_dir.events.write_event(
                    "provider_error", day=day, agent_id=name,
                    error=str(result[1]),
                    tier="negotiation",
                    round=round_num,
                )
                continue

            actions, response = result[1], result[2]
            self.total_tactical_calls += 1
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens

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

            import re as _re_rat
            _rat_m = _re_rat.search(
                r"<rationale>\s*(.*?)\s*</rationale>",
                response.completion, _re_rat.DOTALL,
            )
            rationale_text = _rat_m.group(1).strip() if _rat_m else None

            self.run_dir.events.write_event(
                "agent_turn", day=day,
                agent_id=name,
                tier="tactical",
                round=round_num,
                reasoning=response.completion,
                rationale=rationale_text,
                actions=self._actions_to_dict(actions),
                model=response.model,
                tokens=response.total_tokens,
                latency=response.latency_seconds,
            )

            for flag in self.cot_scanner.scan_reasoning(name, response.completion, day):
                self.run_dir.events.write_event(
                    "cot_flag", day=day,
                    agent_id=flag.agent, tier="tactical",
                    category=flag.category, evidence=flag.evidence,
                    excerpt=flag.excerpt,
                )

            if actions.parse_error:
                self.parse_failures += 1
                continue
            if actions.parse_recovery:
                self.parse_recoveries += 1

            self._execute_actions(
                name, agent, actions, router, day, round_num=round_num,
            )

            has_activity = (
                actions.messages
                or actions.seller_offers
                or actions.buyer_offers
                or actions.accept_offers
                or actions.decline_offers
                or actions.produce_excellent > 0
                or actions.produce_poor > 0
                or actions.build_factory
                or actions.produce_final_goods > 0
                or actions.gossip_posts
            )
            if has_activity:
                self.inactivity.mark_active(name)
                any_action = True

        return any_action

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
        router: MessageRouter, day: int, round_num: int = 0,
    ) -> None:
        """Execute all parsed actions for one agent.

        round_num tags messages with the originating negotiation round
        so within-day delivery can target round R+1 inboxes. The legacy
        single-pass tactical tier passes round_num=0.
        """
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
                    sub_round=round_num,
                )
                agent.record_interaction(day, msg.to, "message_sent")
                # Record on recipient as a received response
                if msg.to in self.agents:
                    self.agents[msg.to].record_interaction(day, name, "response_received")
            except Exception as e:
                self._curr_outcomes[name].append(f"Message to {msg.to}: FAILED ({e})")

        if agent.is_seller:
            # Production FIRST so new widgets are available for offers
            if actions.produce_excellent > 0 or actions.produce_poor > 0:
                try:
                    result = self.market.execute_production(
                        name, excellent=actions.produce_excellent, poor=actions.produce_poor,
                        day=day,
                        defect_rate=float(getattr(self.config.economics, "production_defect_rate", 0.0) or 0.0),
                        rng=self.rng,
                    )
                    self.run_dir.events.write_event("production", day=day, **result)
                    self._curr_outcomes[name].append(
                        f"Production: {result['excellent']}x Excellent, "
                        f"{result['poor']}x Poor -- SUCCESS"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Production: FAILED ({e})")

            # Offers (after production, so new inventory is available)
            # Check protocol-based exclusion (e.g. eBay low reputation)
            if hasattr(self.protocol, "is_excluded") and self.protocol.is_excluded(name, day):
                for offer in actions.seller_offers:
                    self._curr_outcomes[name].append(
                        f"Offer to {offer.to}: BLOCKED (excluded due to low reputation)"
                    )
                actions.seller_offers = []

            for offer in actions.seller_offers:
                try:
                    # Clamp quantity to total inventory available (any
                    # quality). The fulfillment phase at acceptance time
                    # will decide which specific widget to ship.
                    seller_state = self.market.sellers.get(name)
                    available = (
                        sum(seller_state.inventory.values()) if seller_state else 0
                    )
                    qty = min(offer.qty, available)
                    if qty <= 0:
                        self._curr_outcomes[name].append(
                            f"Offer to {offer.to}: SKIPPED (no inventory available)"
                        )
                        continue
                    # Snapshot inventory at placement for post-hoc analysis
                    seller_state = self.market.sellers.get(name)
                    inv_at_placement = (
                        dict(seller_state.inventory) if seller_state
                        else {"Excellent": 0, "Poor": 0}
                    )
                    pending = self.market.place_offer(
                        seller=name, buyer=offer.to,
                        quantity=qty,
                        claimed_quality=offer.claimed_quality,
                        price_per_unit=offer.price_per_unit,
                        day=day,
                        widget_ids=offer.widget_ids or None,
                        claim_rationale=offer.claim_rationale,
                    )
                    evt = self.run_dir.events.write_event(
                        "transaction_proposed", day=day,
                        offer_id=pending.offer_id, seller=name,
                        buyer=offer.to, quantity=qty,
                        claimed_quality=offer.claimed_quality,
                        price_per_unit=offer.price_per_unit,
                        seller_inventory_at_placement=inv_at_placement,
                        committed_widget_ids=list(pending.committed_widget_ids),
                        claim_rationale=pending.claim_rationale,
                    )
                    self._daily_events.setdefault(day, []).append(evt)
                    agent.record_interaction(day, offer.to, "offer_made")
                    self._curr_outcomes[name].append(
                        f"Offer to {offer.to}: {qty}x {offer.claimed_quality} "
                        f"at ${offer.price_per_unit:.2f} -- PLACED (ID: {pending.offer_id})"
                    )
                except Exception as e:
                    self._curr_outcomes[name].append(f"Offer to {offer.to}: FAILED ({e})")

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

        # Protocol-level acceptance gate (redesign 7/8): reputation-aware
        # protocols block acceptance when the price exceeds the EV-derived
        # reservation cap or when the seller is below the gate.
        permit_ok, permit_reason = self.protocol.permit_acceptance(offer, day)
        if not permit_ok:
            self._curr_outcomes[buyer_name].append(
                f"Accept offer {resolved}: AUTO-REFUSED ({permit_reason})"
            )
            self.run_dir.events.write_event(
                "offer_auto_refused", day=day, offer_id=resolved,
                buyer=buyer_name, seller=offer.seller,
                reason=permit_reason,
            )
            return False

        try:
            # Post-redesign: if the offer carries committed_widget_ids,
            # the claim/ship choice was made at PLACEMENT TIME (an
            # explicit choice in the seller's tactical action JSON), and
            # we skip the fulfillment LLM call entirely. The market
            # determines shipped_quality from the actual quality of the
            # committed widgets. This closes the FORCED/CONFABULATED gap.
            shipped_quality = offer.claimed_quality
            widget_ids: list[str] | None = None
            seller_state = self.market.sellers.get(offer.seller)

            if offer.committed_widget_ids and seller_state is not None:
                committed = list(offer.committed_widget_ids)
                by_id = {w.id: w for w in seller_state.widget_instances}
                actual_qs = {by_id[wid].quality for wid in committed if wid in by_id}
                if actual_qs:
                    shipped_quality = next(iter(actual_qs))
                widget_ids = committed
                # Log a fulfillment_decision event for analysis parity with
                # the legacy path. This is now a deterministic record of
                # the seller's commitment, not an LLM call.
                from sanctuary.economics import production_cost as _pc
                factories = max(1, seller_state.factories)
                claimed_cost = _pc(offer.claimed_quality, factories)
                shipped_cost = _pc(shipped_quality, factories)
                cost_differential = round(
                    (claimed_cost - shipped_cost) * offer.quantity, 4,
                )
                evt = self.run_dir.events.write_event(
                    "fulfillment_decision", day=day,
                    seller=offer.seller,
                    buyer=buyer_name,
                    order_id=resolved,
                    quantity=offer.quantity,
                    claimed_quality=offer.claimed_quality,
                    shipped_quality=shipped_quality,
                    widget_ids=widget_ids or [],
                    cost_differential=cost_differential,
                    matched_claim=(shipped_quality == offer.claimed_quality),
                    raw_response="(committed at offer placement)",
                    committed_at_placement=True,
                )
                self._daily_events.setdefault(day, []).append(evt)
            else:
                # Legacy path: fulfillment LLM call decides which widgets ship.
                # Retained for back-compat with offers that pre-date widget-ID
                # commitment (e.g. tests that don't go through the engine).
                fulfillment_enabled = getattr(self.config.run, "fulfillment_phase", True)
                seller_agent = self.agents.get(offer.seller)
                fulfillment_raw: str = ""
                cost_differential = 0.0
                if (fulfillment_enabled
                        and seller_agent is not None
                        and seller_state is not None
                        and seller_state.widget_instances):
                    try:
                        (
                            shipped_quality, widget_ids, fulfillment_raw,
                        ) = seller_agent.fulfillment_call(
                            buyer_name=buyer_name,
                            quantity=offer.quantity,
                            claimed_quality=offer.claimed_quality,
                            price_per_unit=offer.price_per_unit,
                            widget_instances=list(seller_state.widget_instances),
                            revelation_days=REVELATION_LAG_DAYS,
                            current_day=day,
                        )
                        from sanctuary.economics import production_cost as _pc
                        factories = max(1, seller_state.factories)
                        claimed_cost = _pc(offer.claimed_quality, factories)
                        shipped_cost = _pc(shipped_quality, factories)
                        cost_differential = round(
                            (claimed_cost - shipped_cost) * offer.quantity, 4,
                        )
                    except Exception as e:
                        log.warning(
                            "fulfillment_call failed for %s: %s; defaulting to claimed",
                            offer.seller, e,
                        )
                        shipped_quality = offer.claimed_quality
                        widget_ids = None
                        fulfillment_raw = f"ERROR: {e}"

                    evt = self.run_dir.events.write_event(
                        "fulfillment_decision", day=day,
                        seller=offer.seller,
                        buyer=buyer_name,
                        order_id=resolved,
                        quantity=offer.quantity,
                        claimed_quality=offer.claimed_quality,
                        shipped_quality=shipped_quality,
                        widget_ids=widget_ids or [],
                        cost_differential=cost_differential,
                        matched_claim=(shipped_quality == offer.claimed_quality),
                        raw_response=fulfillment_raw[:2000],
                        committed_at_placement=False,
                    )
                    self._daily_events.setdefault(day, []).append(evt)

            revelation_day = self.revelation_scheduler.schedule(
                transaction_id=resolved,
                seller=offer.seller,
                buyer=buyer_name,
                claimed_quality=offer.claimed_quality,
                true_quality=shipped_quality,
                quantity=offer.quantity,
                transaction_day=day,
            )

            tx = self.market.accept_offer(
                resolved, revelation_day, day,
                shipped_quality=shipped_quality, widget_ids=widget_ids,
            )

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

    # ── Memory consolidation helpers (spec §7) ──────────────────────────────

    def _memory_inputs_for(self, name: str, day: int) -> dict[str, str]:
        """Compute the memory-consolidation injections for one tactical call:
        yesterday's summary, the performance ledger, the strategic-memo
        digest, and the structured living ledger.

        living_ledger replaces metric_ledger as the primary state-of-the-firm
        block (cash trajectory + inventory with widget IDs + production
        history + offer outcomes). The narrative metric_ledger is still
        passed for back-compat.
        """
        from sanctuary.living_ledger import build_living_ledger

        agent = self.agents.get(name)
        if agent is None:
            return {
                "prev_day_summary": "",
                "metric_ledger": "",
                "strategic_digest": "",
                "living_ledger": "",
            }

        prev_day_summary = self._daily_summaries.get(name, {}).get(day - 1, "")

        is_seller = agent.is_seller
        state = (
            self.market.sellers.get(name) if is_seller
            else self.market.buyers.get(name)
        )
        metric_ledger = build_metric_ledger(
            name=name,
            is_seller=is_seller,
            state=state,
            transactions=self.market.transactions,
            daily_events=self._daily_events,
            current_day=day,
        )
        strategic_digest = digest_recent_memos(agent.policy_history, k=10)
        living_ledger = build_living_ledger(
            name=name,
            market=self.market,
            day=day,
            daily_events=self._daily_events,
        )
        return {
            "prev_day_summary": prev_day_summary,
            "metric_ledger": metric_ledger,
            "strategic_digest": strategic_digest,
            "living_ledger": living_ledger,
        }

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
                lines.append(f"  {name}: ${b['cash']:,.2f}, {b.get('widget_inventory', 0)} widgets")
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
            d: dict[str, Any] = {
                "accept_offers": actions.accept_offers,
                "decline_offers": actions.decline_offers,
            }
        else:
            d = {
                "messages": [{"to": m.to, "public": m.public, "body": m.body} for m in actions.messages],
                "seller_offers": [
                    {"to": o.to, "qty": o.qty, "claimed_quality": o.claimed_quality,
                     "price_per_unit": o.price_per_unit}
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
        if actions.parse_error:
            d["parse_error"] = actions.parse_error
        if actions.parse_recovery:
            d["parse_recovery"] = actions.parse_recovery
        return d

    def _broadcast_state(self) -> None:
        """Send state to dashboard if connected."""
        if self._dashboard_broadcast:
            try:
                self._dashboard_broadcast(self.market.daily_snapshot())
            except Exception:
                pass
