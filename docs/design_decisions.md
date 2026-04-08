# Design Decisions — The Sanctuary

A running log of non-obvious design choices, empirical failures, and the reasoning
behind revisions. Intended for writeup reference.

---

## 2026-04-08 — Reversion from downstream-manufacturer model to quota system

### What we built first (and why)

The initial buyer model was a **downstream-manufacturer model**: buyers purchase
widgets, transform them into final goods, and earn revenue based on the true quality
of the input (revealed after a stochastic delay). Buyers faced a fixed daily
operating cost ($40/day) to create participation pressure, but participation itself
was optional — a buyer could simply decline all offers and pay the fixed cost as an
"idling fee."

The rationale for this design was theoretical elegance:

1. **Captive-demand realism.** A buyer who *must* buy any specific quantity feels
   like a contrived setup. A downstream manufacturer who can choose to idle (at
   cost) felt more realistic — it mirrors a firm that can reduce output when
   input costs are prohibitive.

2. **Richer strategic space.** Optional participation was supposed to give buyers
   a credible threat point in price negotiations: "price too high and I'll idle."
   This seemed like it would generate interesting price-pressure dynamics.

3. **Credence-good mechanics.** The final-good revenue structure (dependent on
   revealed true quality) was retained from an earlier codebase and preserved
   the core misrepresentation-risk mechanic we wanted to study.

### What the data showed (the frozen-market result)

The 30-day run under the downstream-manufacturer model is preserved as a negative
control in `runs/baseline_no_quota_frozen_market/`.

Key findings:
- All four buyers independently set price ceilings of ~$35 for Excellent and ~$18
  for Poor widgets.
- All four sellers set floors of ~$75–$95 for Excellent and ~$40–$60 for Poor.
- **Zero transactions occurred across the entire 30-day run.**
- All buyer cash balances ended at exactly $4,800 (= $6,000 − $40 × 30), confirming
  they sat out every day.

The buyers' reasoning was individually rational. At final-good prices of ~$90
(Excellent) and ~$52 (Poor), with a 4-unit daily cap and $40 fixed cost, the
break-even price per Excellent widget is:
  ($90 − $40/4) ≈ $80 at cap, or ($90 − $40) ≈ $50 at one unit.

With no misrepresentation yet observed, a conservative LLM agent demanding safety
margin against deception risk sets a ceiling well below seller floors. The $40 fixed
cost was small enough that sitting out was always preferable to buying at prices
the agents considered "too high." The fixed cost created pressure in principle but
not in practice for LLM agents with defensive priors.

### Why the downstream model failed for LLM agents specifically

The root cause is a **degenerate equilibrium problem with LLM agents under optional
participation**. Classical economic theory assumes agents will eventually accept
trades when the cost of not trading exceeds the cost of trading at current prices.
LLM agents do not converge to this equilibrium because:

1. **Defensive priors dominate.** LLM agents, trained on cautious
   human text, default to "wait for better conditions" when participation is
   optional. They interpret the fixed cost as an acceptable status-quo cost,
   not as a penalty that should force participation.

2. **No learning from inaction.** With 0 transactions across 30 days, agents
   never received feedback that their strategy was failing. The strategic tier
   saw no revelations, no price history, no evidence of competitor trades — so
   it kept recommending the same conservative ceilings.

3. **The optional-participation equilibrium is stable for LLMs.** Once an agent
   sets a ceiling below seller floors, it has no reason to revise upward — there
   is no direct cost signal (aside from the small fixed cost) indicating that
   non-participation is expensive.

### What we changed

We reverted to a **quota system** adapted from an earlier codebase, retaining the
final-good mechanics for the credence-good dynamic:

- Each buyer must acquire **30 widgets over 30 days** (1/day on average).
- **Daily flow penalty:** $2/day per unfulfilled quota unit. A buyer who has
  acquired 10 widgets by day 15 owes $40/day in penalties (20 units × $2).
- **Terminal penalty:** $60/unit unfulfilled at end of day 30.
- **Worst-case cost of non-participation:** $2 × 30 × 30 + $60 × 30 = **$3,600**
  — more than half of buyer starting cash ($6,000).
- The $40/day fixed cost is removed. The penalty system provides the pressure.

The final-good production mechanic is retained unchanged: buyers still convert
acquired widgets into revenue based on TRUE quality (revealed with delay), so
misrepresentation still has real economic consequences.

### What this teaches about LLM agent incentive design

> **Optional participation creates degenerate equilibria with current LLM agents.**

When participation is voluntary and the cost of non-participation is low relative
to the perceived risk of participating, LLM agents default to inaction. This is not
a failure of reasoning — it is a consequence of the agents' training distribution
(cautious human deliberation) applied to a setting where the designer intended
economic pressure to force participation.

Design principle extracted: **Explicit, quantified participation penalties are
necessary to produce economic activity in LLM agent markets.** The quota system
works because it makes the cost of non-participation *salient in the prompt* at
every decision point (the tactical prompt shows daily penalty and quota remaining),
not just because the math works out.

This finding is likely generalizable: LLM agents will not "discover" equilibria
that require trading away an option value. Incentive structures must be designed
with agent prompt visibility in mind, not just with economically correct math.

### Reference

Frozen-market baseline: `runs/baseline_no_quota_frozen_market/`
Quota-model first run: `runs/run_20260408_054618_seed42/`

---

## 2026-04-08 — Silent action failures, hallucinated state, and structural fixes

### What the second run showed

The quota-model run (`run_20260408_054618_seed42`) produced **0 transactions across 30 days**
despite agents reasoning correctly about the quota incentives. The root cause was a single
string-truncation bug: offer IDs were displayed to agents as `[f439fd43…]` (8 hex chars +
ellipsis), and agents faithfully copied that truncated form into their `accept_offers` arrays.
The simulation looked up `pending_offers.get("f439fd43")`, got `None`, and silently returned.
Every accept in the entire run was a no-op.

Because the simulation gave no feedback on failed accepts, agents inferred their accepts had
succeeded — they never received any signal contradicting that inference. By day 8, Halcyon
Assembly's strategic memo stated "QUOTA COMPLETE: We acquired all 30 widgets by day 6" — a
complete hallucination based on correct reasoning applied to incomplete information. Once formed,
this false belief persisted for 22 days because nothing in the prompt context ever contradicted it.

### What this teaches about LLM agent system design

> **Silent action failures are catastrophic for agent systems. Always surface action outcomes
> in the next turn's context. Always provide an authoritative state header that does not depend
> on agent memory. The frozen-market run of 2026-04-08 demonstrated how a single
> string-truncation bug can cascade into 30-day false belief states across 8 agents because
> nothing in the prompt context contradicted the agents' incorrect inferences.**

Three structural fixes were applied:

**Fix 1 — Stop truncating offer IDs.** `format_pending_offers_for_buyer()` and
`format_pending_offers_for_seller()` now show full UUIDs. A prefix-match fallback was added to
`_try_accept_offer()`: if the exact ID is not found, scan for a unique pending offer whose ID
starts with the provided string. Ambiguous or zero matches produce a logged `accept_attempted_failed`
event and a `✗` outcome line — transparent failure instead of silent failure.

**Fix 2 — Surface action results in the next turn.** Every tactical prompt now opens with a
`[YOUR PREVIOUS TURN OUTCOMES — Day N-1]` section listing every action outcome from the prior day:
`✓` for successes (offer placed, offer accepted, production succeeded), `✗` for failures
(accept failed, production rejected, offer expired). An agent that submits an accept that fails
will see `✗ Accept failed: tried to accept offer "..." — no matching offer found` at the top of
its very next prompt. False beliefs about action success are structurally impossible when every
failure is surfaced.

**Fix 3 — Authoritative state header.** Every tactical and strategic prompt now opens with a
`[YOUR CURRENT STATE — Start of Day N]` section drawn directly from `MarketState` for that agent:
exact cash, inventory with revelation status, quota progress, accrued penalties, factory count.
This is not inferred from conversation history — it is the ground truth. Even if an agent's prior
turn was completely confused, the state header pulls it back to reality.

### Design principles extracted

1. **Agents copy what they see.** If the prompt shows `f439fd43…`, the agent outputs `f439fd43…`.
   Never truncate identifiers that agents need to use as keys. Show the full string.

2. **Silence is not a valid error mode.** Every action that can fail must produce visible feedback
   in the next turn. Silent failures propagate into false beliefs that compound over time.

3. **Ground-truth headers beat memory.** Agent conversation history is an unreliable source of
   state, especially when actions silently fail. An authoritative state block drawn from simulation
   state — not from what the agent thinks happened — eliminates an entire class of hallucination.

4. **These principles generalize.** Any LLM agent system that relies on agents tracking their own
   state via conversation history, without authoritative external grounding, will exhibit these
   failure modes under adversarial conditions (bugs, rate limits, truncation, any silent error).

### Reference

Second run (0 transactions, UUID bug): `runs/run_20260408_054618_seed42/`
Structural fix commit: see git log for commit "Fix truncated offer IDs, add prev-turn outcomes, add authoritative state header"
