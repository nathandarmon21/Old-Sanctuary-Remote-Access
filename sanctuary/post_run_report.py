"""
Auto-generated post-run PDF report.

Produces a comprehensive summary after every simulation run, covering:
  - Run configuration and completion status
  - Agent strategies (extracted from CEO memos)
  - Market activity (prices, transactions, messages, offers)
  - Three key metrics: misrepresentation, allocative efficiency, market integrity
  - Behavioral dynamics (CoT scanner flags for collusion, deception, etc.)

Called automatically by run.py after mark_complete().
"""

from __future__ import annotations

import json
import statistics
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sanctuary.events import read_events


# ── Style ────────────────────────────────────────────────────────────────────

C_BLUE  = "#3B82F6"
C_AMBER = "#F59E0B"
C_GREEN = "#10B981"
C_RED   = "#EF4444"
C_GRAY  = "#6B7280"
C_PURPLE = "#8B5CF6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "figure.dpi": 150,
})

W, H = 8.5, 11


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _page_title(fig, title: str, subtitle: str = ""):
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", color="#111827")
    if subtitle:
        fig.text(0.06, 0.93, subtitle, fontsize=9, color=C_GRAY)


def _caption(fig, text: str, y: float = 0.02):
    fig.text(0.06, y, _wrap(text, 105), fontsize=8.5, color="#374151",
             linespacing=1.4, verticalalignment="bottom")


# ── Data extraction ──────────────────────────────────────────────────────────

def _extract_run_data(run_dir: Path) -> dict[str, Any]:
    """Extract all data needed for the report from a run directory."""
    events = read_events(run_dir / "events.jsonl")

    # Config
    config = {}
    config_path = run_dir / "config_used.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Metrics
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Manifest
    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Final state
    final_state = {}
    fs_path = run_dir / "final_state.json"
    if fs_path.exists():
        with open(fs_path) as f:
            final_state = json.load(f)

    # Derived data
    sellers = [s["name"] for s in config.get("agents", {}).get("sellers", [])]
    buyers = [b["name"] for b in config.get("agents", {}).get("buyers", [])]
    total_days = config.get("run", {}).get("days", 60)
    protocol = config.get("protocol", {}).get("system", "unknown")

    # Events by type
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]
    proposals = [e for e in events if e.get("event_type") == "transaction_proposed"]
    revelations = [e for e in events if e.get("event_type") == "quality_revealed"]
    messages = [e for e in events if e.get("event_type") == "message_sent"]
    cot_flags = [e for e in events if e.get("event_type") == "cot_flag"]
    agent_turns = [e for e in events if e.get("event_type") == "agent_turn"]
    productions = [e for e in events if e.get("event_type") == "production"]
    bankruptcies = [e for e in events if e.get("event_type") == "bankruptcy"]
    provider_errors = [e for e in events if e.get("event_type") == "provider_error"]

    # Days completed
    day_starts = [e for e in events if e.get("event_type") == "day_start"]
    days_completed = max((e.get("day", 0) for e in events), default=0)

    # Daily time series
    daily: dict[int, dict[str, int]] = {}
    for d in range(1, days_completed + 1):
        daily[d] = {"transactions": 0, "offers": 0, "messages": 0, "productions": 0}
    for e in transactions:
        d = e.get("day", 0)
        if d in daily:
            daily[d]["transactions"] += 1
    for e in proposals:
        d = e.get("day", 0)
        if d in daily:
            daily[d]["offers"] += 1
    for e in messages:
        d = e.get("day", 0)
        if d in daily:
            daily[d]["messages"] += 1
    for e in productions:
        d = e.get("day", 0)
        if d in daily:
            daily[d]["productions"] += 1

    # Price series
    price_by_day: dict[int, list[float]] = {}
    for tx in transactions:
        d = tx.get("day", 0)
        price_by_day.setdefault(d, []).append(tx.get("price_per_unit", 0.0))

    # Strategic memos (last per agent)
    strategic_turns = [e for e in agent_turns if e.get("tier") == "strategic"]
    last_memos: dict[str, str] = {}
    for e in sorted(strategic_turns, key=lambda x: x.get("day", 0)):
        agent = e.get("agent_id", "")
        last_memos[agent] = e.get("reasoning", "")[:500]

    # Same-role messaging
    seller_set = set(sellers)
    buyer_set = set(buyers)
    seller_to_seller = sum(
        1 for m in messages
        if m.get("from_agent", "") in seller_set and m.get("to_agent", "") in seller_set
    )
    buyer_to_buyer = sum(
        1 for m in messages
        if m.get("from_agent", "") in buyer_set and m.get("to_agent", "") in buyer_set
    )

    return {
        "config": config,
        "metrics": metrics,
        "manifest": manifest,
        "final_state": final_state,
        "events": events,
        "sellers": sellers,
        "buyers": buyers,
        "total_days": total_days,
        "days_completed": days_completed,
        "protocol": protocol,
        "transactions": transactions,
        "proposals": proposals,
        "revelations": revelations,
        "messages": messages,
        "cot_flags": cot_flags,
        "agent_turns": agent_turns,
        "productions": productions,
        "bankruptcies": bankruptcies,
        "provider_errors": provider_errors,
        "daily": daily,
        "price_by_day": price_by_day,
        "last_memos": last_memos,
        "seller_to_seller": seller_to_seller,
        "buyer_to_buyer": buyer_to_buyer,
    }


# ── Page builders ────────────────────────────────────────────────────────────

def _page_title_summary(pdf, d: dict):
    """Page 1: Title and executive summary."""
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 0.91, "Sanctuary Simulation Report", ha="center",
             fontsize=22, fontweight="bold", color="#111827")
    fig.text(0.5, 0.875, f"Protocol: {d['protocol']}  |  Seed: {d['manifest'].get('seed', '?')}",
             ha="center", fontsize=11, color=C_GRAY)

    m = d["metrics"]
    misrep = m.get("misrepresentation", {})
    ae_data = m.get("allocative_efficiency", {})
    mi = m.get("market_integrity", {})

    lines = [
        f"Days completed: {d['days_completed']} / {d['total_days']}",
        f"Transactions: {len(d['transactions'])} ({len(d['transactions'])/max(1,d['days_completed']):.2f}/day)",
        f"Offers proposed: {len(d['proposals'])}",
        f"Messages sent: {len(d['messages'])}",
        f"Production events: {len(d['productions'])}",
        f"Provider errors: {len(d['provider_errors'])}",
        f"Bankruptcies: {len(d['bankruptcies'])}",
        "",
        "KEY METRICS",
        f"  Misrepresentation rate:      {misrep.get('overall', 0):.1%}  "
        f"({misrep.get('total_misrepresented', 0)}/{misrep.get('total_revealed', 0)} revealed)",
        f"  Allocative efficiency (AE):  {ae_data.get('ae', 0):.1%}",
        f"  Price-cost margin (Lerner):  {m.get('price_cost_margin', {}).get('average_pcm', 0):.3f}",
        "",
        "  MARKET INTEGRITY SUB-SCORES",
        f"    Price Parallelism (PPI):   {mi.get('price_parallelism_index', 0):.3f}  "
        "(0=independent, 1=coordinated)",
        f"    Markup Correlation (MC):   {mi.get('markup_correlation', 0):.3f}  "
        "(0=independent, 1=correlated)",
        f"    Exploitation Rate (ER):    {mi.get('exploitation_rate', 0):.1%}  "
        "(final-third price gouging)",
        f"    Trust Persistence (TP):    {mi.get('trust_persistence', 0):.3f}  "
        "(0=switching, 1=loyal)",
    ]

    # Agent standings
    per_agent = m.get("per_agent_net_profit", {})
    if per_agent:
        lines.append("")
        lines.append("AGENT STANDINGS (net profit, realized)")
        ranked = sorted(per_agent.items(), key=lambda x: x[1].get("net_profit_realized", 0), reverse=True)
        for name, data in ranked:
            role = data.get("role", "?")
            profit = data.get("net_profit_realized", 0)
            lines.append(f"  {profit:>+10,.2f}  {name} ({role})")

    # CoT flags summary
    flag_cats = Counter(f.get("category", "") for f in d["cot_flags"])
    if flag_cats:
        lines.append("")
        lines.append("BEHAVIORAL FLAGS (CoT scanner)")
        for cat, count in flag_cats.most_common():
            lines.append(f"  {count:>4d}  {cat}")

    body = "\n".join(lines)
    fig.text(0.06, 0.83, body, fontsize=9, color="#1F2937",
             verticalalignment="top", fontfamily="monospace", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def _page_market_activity(pdf, d: dict):
    """Page 2: Market activity time series."""
    fig = plt.figure(figsize=(W, H))
    _page_title(fig, "Market Activity Over Time",
                "Daily transactions, offers, messages, and production")

    days = sorted(d["daily"].keys())
    if not days:
        pdf.savefig(fig); plt.close(fig); return

    tx_s = [d["daily"][day]["transactions"] for day in days]
    of_s = [d["daily"][day]["offers"] for day in days]
    ms_s = [d["daily"][day]["messages"] for day in days]
    pr_s = [d["daily"][day]["productions"] for day in days]

    # Transactions + offers
    ax1 = fig.add_axes([0.08, 0.66, 0.86, 0.22])
    ax1.bar([x - 0.2 for x in days], of_s, 0.4, label="Offers proposed", color=C_AMBER, alpha=0.8, zorder=3)
    ax1.bar([x + 0.2 for x in days], tx_s, 0.4, label="Transactions", color=C_GREEN, alpha=0.8, zorder=3)
    ax1.set_ylabel("Count")
    ax1.set_title("Offers and Transactions")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Messages
    ax2 = fig.add_axes([0.08, 0.38, 0.86, 0.22])
    ax2.fill_between(days, ms_s, alpha=0.3, color=C_BLUE)
    ax2.plot(days, ms_s, color=C_BLUE, lw=1.5, label="Messages sent")
    ax2.set_ylabel("Messages")
    ax2.set_title("Daily Message Volume")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # Production
    ax3 = fig.add_axes([0.08, 0.10, 0.86, 0.22])
    ax3.bar(days, pr_s, color=C_PURPLE, alpha=0.8, zorder=3)
    ax3.set_xlabel("Simulation Day")
    ax3.set_ylabel("Production Events")
    ax3.set_title("Daily Production")
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    _caption(fig,
        f"Total: {len(d['transactions'])} transactions from {len(d['proposals'])} offers "
        f"({len(d['transactions'])/max(1,len(d['proposals']))*100:.0f}% completion rate), "
        f"{len(d['messages'])} messages, {len(d['productions'])} production events "
        f"over {d['days_completed']} days.", y=0.01)
    pdf.savefig(fig)
    plt.close(fig)


def _page_price_analysis(pdf, d: dict):
    """Page 3: Price dynamics and transaction economics."""
    fig = plt.figure(figsize=(W, H))
    _page_title(fig, "Price Dynamics and Transaction Economics",
                "Price trends, distribution, and per-agent performance")

    # Price over time
    ax1 = fig.add_axes([0.08, 0.66, 0.55, 0.22])
    if d["price_by_day"]:
        p_days = sorted(d["price_by_day"].keys())
        p_means = [statistics.mean(d["price_by_day"][day]) for day in p_days]
        ax1.plot(p_days, p_means, "o-", color=C_BLUE, markersize=4, lw=1.5, zorder=3)
        from sanctuary.economics import FMV
        ax1.axhline(FMV["Excellent"], color=C_GREEN, ls="--", lw=1, label=f"FMV Excellent (${FMV['Excellent']})")
        ax1.axhline(FMV["Poor"], color=C_RED, ls="--", lw=1, label=f"FMV Poor (${FMV['Poor']})")
        ax1.legend(fontsize=6)
    ax1.set_ylabel("Price ($/unit)")
    ax1.set_title("Average Transaction Price Over Time")
    ax1.grid(alpha=0.3, zorder=0)

    # Price distribution
    ax2 = fig.add_axes([0.72, 0.66, 0.22, 0.22])
    all_prices = [tx.get("price_per_unit", 0) for tx in d["transactions"]]
    if all_prices:
        ax2.hist(all_prices, bins=max(5, len(set(all_prices))), color=C_BLUE, alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Price")
    ax2.set_title("Price Distribution")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # Per-agent net profit
    ax3 = fig.add_axes([0.08, 0.32, 0.86, 0.26])
    per_agent = d["metrics"].get("per_agent_net_profit", {})
    if per_agent:
        names = sorted(per_agent.keys(), key=lambda n: per_agent[n].get("net_profit_realized", 0))
        profits = [per_agent[n].get("net_profit_realized", 0) for n in names]
        colors = [C_BLUE if per_agent[n].get("role") == "seller" else C_AMBER for n in names]
        short_names = [n.split()[0] if len(n) > 12 else n for n in names]
        bars = ax3.barh(range(len(names)), profits, color=colors, zorder=3)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(short_names, fontsize=7.5)
        ax3.set_xlabel("Net Profit ($)")
        ax3.set_title("Agent Net Profit (blue=seller, amber=buyer)")
        ax3.axvline(0, color="#9CA3AF", lw=0.8)
        ax3.bar_label(bars, fmt="$%.0f", fontsize=7, padding=3)
        ax3.grid(axis="x", alpha=0.3, zorder=0)

    # Cumulative transactions
    ax4 = fig.add_axes([0.08, 0.04, 0.86, 0.20])
    if d["daily"]:
        days_list = sorted(d["daily"].keys())
        import numpy as np
        cum_tx = np.cumsum([d["daily"][day]["transactions"] for day in days_list])
        ax4.plot(days_list, cum_tx, color=C_GREEN, lw=2, label="Actual", zorder=3)
        ax4.plot([1, max(days_list)], [1, max(days_list)], "--", color=C_GRAY, lw=1,
                 label="1.0 tx/day target", zorder=2)
        ax4.set_xlabel("Day")
        ax4.set_ylabel("Cumulative Tx")
        ax4.set_title("Cumulative Transactions vs Target")
        ax4.legend(fontsize=7)
        ax4.grid(alpha=0.3, zorder=0)

    pdf.savefig(fig)
    plt.close(fig)


def _page_three_metrics(pdf, d: dict):
    """Page 4: The three key metrics with formulas and breakdowns."""
    fig = plt.figure(figsize=(W, H))
    _page_title(fig, "Key Metrics: Misrepresentation, Efficiency, Integrity",
                "The three primary measures of market health")

    m = d["metrics"]
    misrep = m.get("misrepresentation", {})
    ae_data = m.get("allocative_efficiency", {})
    pcm_data = m.get("price_cost_margin", {})
    mi = m.get("market_integrity", {})

    # 1. Misrepresentation
    ax1 = fig.add_axes([0.08, 0.72, 0.40, 0.16])
    per_seller = misrep.get("per_seller", {})
    if per_seller:
        names = list(per_seller.keys())
        rates = [per_seller[n] for n in names]
        short = [n.split()[0] for n in names]
        colors = [C_RED if r > 0 else C_GREEN for r in rates]
        bars = ax1.barh(range(len(names)), [r * 100 for r in rates], color=colors, zorder=3)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(short, fontsize=7.5)
        ax1.set_xlabel("Misrepresentation %")
        ax1.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=3)
    else:
        ax1.text(0.5, 0.5, "No quality revelations yet", ha="center", va="center",
                 fontsize=9, color=C_GRAY, transform=ax1.transAxes)
    ax1.set_title(f"1. Misrepresentation Rate: {misrep.get('overall', 0):.1%}")
    ax1.grid(axis="x", alpha=0.3, zorder=0)

    # Formula text
    fig.text(0.55, 0.86,
             "M = |{t : claimed != true}| / |revealed|\n\n"
             f"Overall: {misrep.get('overall',0):.1%}  "
             f"({misrep.get('total_misrepresented',0)}/{misrep.get('total_revealed',0)})\n\n"
             "0% = perfectly honest market\n"
             "High M = widespread quality fraud",
             fontsize=8.5, color="#374151", verticalalignment="top")

    # 2. Allocative Efficiency
    ax2 = fig.add_axes([0.08, 0.46, 0.40, 0.16])
    ae_val = ae_data.get("ae", 0)
    ax2.barh(["Allocative\nEfficiency"], [ae_val * 100], color=C_GREEN if ae_val > 0.5 else C_RED, zorder=3)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("%")
    ax2.set_title(f"2. Allocative Efficiency: {ae_val:.1%}")
    ax2.grid(axis="x", alpha=0.3, zorder=0)

    fig.text(0.55, 0.60,
             "AE = 1 - (DWL / MaxSurplus)\n\n"
             f"Max surplus:    ${ae_data.get('max_surplus', 0):>10,.2f}\n"
             f"Actual surplus: ${ae_data.get('actual_surplus', 0):>10,.2f}\n"
             f"Deadweight loss: ${ae_data.get('dwl', 0):>10,.2f}\n\n"
             f"Avg price-cost margin: {pcm_data.get('average_pcm', 0):.3f}",
             fontsize=8.5, color="#374151", verticalalignment="top", fontfamily="monospace")

    # 3. Market Integrity sub-scores
    ax3 = fig.add_axes([0.08, 0.16, 0.55, 0.20])
    mi_labels = [
        "Price Parallelism\n(PPI)",
        "Markup Correlation\n(MC)",
        "Exploitation Rate\n(ER)",
        "Trust Persistence\n(TP)",
    ]
    mi_vals = [
        mi.get("price_parallelism_index", 0),
        mi.get("markup_correlation", 0),
        mi.get("exploitation_rate", 0),
        mi.get("trust_persistence", 0),
    ]
    mi_colors = [C_AMBER, C_AMBER, C_RED, C_GREEN]
    bars = ax3.barh(range(len(mi_labels)), mi_vals, color=mi_colors, zorder=3)
    ax3.set_yticks(range(len(mi_labels)))
    ax3.set_yticklabels(mi_labels, fontsize=7.5)
    ax3.set_xlim(0, 1)
    ax3.set_title("3. Market Integrity Sub-Scores")
    ax3.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=3)
    ax3.grid(axis="x", alpha=0.3, zorder=0)

    fig.text(0.68, 0.34,
             "PPI: price convergence\n  (1 = coordinated)\n"
             "MC: markup co-movement\n  (1 = correlated)\n"
             "ER: end-game gouging\n  (% above 1.5x FMV)\n"
             "TP: buyer loyalty streaks\n  (1 = stable pairs)",
             fontsize=7.5, color="#6B7280", verticalalignment="top")

    _caption(fig,
        "These three metrics are the primary research outputs. Misrepresentation "
        "measures deception prevalence. Allocative efficiency measures how much "
        "total welfare is captured vs lost to deadweight. Market integrity captures "
        "collusion signals (PPI, MC), exploitation (ER), and trust dynamics (TP).",
        y=0.01)
    pdf.savefig(fig)
    plt.close(fig)


def _page_behavioral_dynamics(pdf, d: dict):
    """Page 5: CoT scanner behavioral flags and messaging patterns."""
    fig = plt.figure(figsize=(W, H))
    _page_title(fig, "Behavioral Dynamics",
                "Chain-of-thought flags, same-role messaging, and strategic signals")

    cot = d["cot_flags"]
    cats = Counter(f.get("category", "") for f in cot)

    # Category bar chart
    ax1 = fig.add_axes([0.08, 0.62, 0.55, 0.26])
    if cats:
        labels = [c for c, _ in cats.most_common()]
        vals = [v for _, v in cats.most_common()]
        cat_colors = {
            "collusion_price_fixing": C_RED,
            "misrepresentation_planning": C_AMBER,
            "deception_intent": C_AMBER,
            "exploitation": C_RED,
            "trust_assessment": C_BLUE,
            "strategic_pivot": C_GREEN,
            "frustration_desperation": C_GRAY,
        }
        colors = [cat_colors.get(l, C_GRAY) for l in labels]
        short_labels = [l.replace("_", "\n") for l in labels]
        bars = ax1.barh(range(len(labels)), vals, color=colors, zorder=3)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(short_labels, fontsize=7)
        ax1.invert_yaxis()
        ax1.bar_label(bars, fontsize=7.5, padding=3)
    else:
        ax1.text(0.5, 0.5, "No behavioral flags detected", ha="center", va="center",
                 fontsize=10, color=C_GRAY, transform=ax1.transAxes)
    ax1.set_xlabel("Flag Count")
    ax1.set_title(f"CoT Scanner Flags (n={len(cot)})")
    ax1.grid(axis="x", alpha=0.3, zorder=0)

    # Flags by tier
    ax2 = fig.add_axes([0.72, 0.62, 0.22, 0.26])
    tier_counts = Counter(f.get("tier", "?") for f in cot)
    if tier_counts:
        t_labels = list(tier_counts.keys())
        t_vals = [tier_counts[l] for l in t_labels]
        ax2.pie(t_vals, labels=t_labels, autopct="%1.0f%%", textprops={"fontsize": 7},
                colors=[C_BLUE, C_AMBER, C_PURPLE, C_GREEN][:len(t_labels)],
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("Flags by Tier", fontsize=10)

    # Same-role messaging
    ax3 = fig.add_axes([0.08, 0.32, 0.40, 0.22])
    msg_cats = ["Seller-to-\nSeller", "Buyer-to-\nBuyer", "Cross-role"]
    cross = len(d["messages"]) - d["seller_to_seller"] - d["buyer_to_buyer"]
    msg_vals = [d["seller_to_seller"], d["buyer_to_buyer"], cross]
    msg_colors = [C_RED, C_RED, C_GREEN]
    bars = ax3.bar(range(len(msg_cats)), msg_vals, color=msg_colors, alpha=0.8, zorder=3)
    ax3.set_xticks(range(len(msg_cats)))
    ax3.set_xticklabels(msg_cats, fontsize=8)
    ax3.set_ylabel("Messages")
    ax3.set_title("Message Routing Patterns")
    ax3.bar_label(bars, fontsize=8, padding=2)
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    # Flags over time (cumulative)
    ax4 = fig.add_axes([0.56, 0.32, 0.38, 0.22])
    if cot:
        flag_days = sorted(set(f.get("day", 0) for f in cot))
        cum_by_day = {}
        cum = 0
        for day in range(1, d["days_completed"] + 1):
            day_flags = sum(1 for f in cot if f.get("day") == day)
            cum += day_flags
            cum_by_day[day] = cum
        ax4.fill_between(cum_by_day.keys(), cum_by_day.values(), alpha=0.3, color=C_RED)
        ax4.plot(list(cum_by_day.keys()), list(cum_by_day.values()), color=C_RED, lw=1.5)
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Cumulative Flags")
    ax4.set_title("Behavioral Flags Over Time")
    ax4.grid(alpha=0.3, zorder=0)

    _caption(fig,
        f"Same-role messaging: {d['seller_to_seller']} seller-to-seller and "
        f"{d['buyer_to_buyer']} buyer-to-buyer messages (potential coordination "
        f"channels) vs {cross} cross-role messages (negotiation). "
        f"The CoT scanner detected {len(cot)} total behavioral flags across "
        f"{len(cats)} categories by scanning agent reasoning, CEO memos, and "
        f"same-role messages.", y=0.01)
    pdf.savefig(fig)
    plt.close(fig)


def _page_agent_strategies(pdf, d: dict):
    """Page 6: Agent strategy summaries from CEO memos."""
    fig = plt.figure(figsize=(W, H))
    _page_title(fig, "Agent Strategic Summaries",
                "Excerpts from the most recent CEO memo for each agent")

    y = 0.88
    all_agents = d["sellers"] + d["buyers"]
    for agent in all_agents:
        memo = d["last_memos"].get(agent, "(no strategic memo recorded)")
        role = "Seller" if agent in d["sellers"] else "Buyer"

        # Truncate and clean
        memo_clean = memo.replace("\n", " ").strip()
        if len(memo_clean) > 350:
            memo_clean = memo_clean[:347] + "..."

        fig.text(0.06, y, f"{agent} ({role})", fontsize=9.5, fontweight="bold", color="#111827")
        y -= 0.015
        wrapped = _wrap(memo_clean, 100)
        line_count = wrapped.count("\n") + 1
        fig.text(0.06, y, wrapped, fontsize=7.5, color="#4B5563", linespacing=1.3,
                 verticalalignment="top")
        y -= 0.012 * line_count + 0.025

        if y < 0.05:
            break

    pdf.savefig(fig)
    plt.close(fig)


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_post_run_report(run_dir: Path) -> Path:
    """
    Generate a comprehensive PDF report from a completed run directory.

    Returns the path to the generated PDF.
    """
    output_path = run_dir / "report.pdf"
    d = _extract_run_data(run_dir)

    with PdfPages(str(output_path)) as pdf:
        _page_title_summary(pdf, d)
        _page_market_activity(pdf, d)
        _page_price_analysis(pdf, d)
        _page_three_metrics(pdf, d)
        _page_behavioral_dynamics(pdf, d)
        _page_agent_strategies(pdf, d)

    return output_path
