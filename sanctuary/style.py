"""
Canonical visual style for the Sanctuary simulation.

Defines the warm pastel palette used in both:
  - matplotlib plots (PDF report)
  - dashboard HTML/CSS

All plot code and dashboard code should import from here so that
the visual identity stays consistent and can be tuned in one place.

Palette philosophy: a natural history field journal rather than a
corporate finance report. Soft cream backgrounds, muted earth tones
and warm pastels, no saturated primaries.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Colour palette ────────────────────────────────────────────────────────────

# Page / canvas background
BACKGROUND: Final[str] = "#FAF6F0"   # warm cream

# Grid lines (very light)
GRID_COLOR: Final[str] = "#E8E0D5"   # pale grey-brown

# Axes background (same as page)
AXES_BACKGROUND: Final[str] = "#FAF6F0"

# Text colours
TEXT_PRIMARY: Final[str] = "#2E2A25"    # deep warm charcoal
TEXT_SECONDARY: Final[str] = "#6B6257"  # muted warm grey

# Accent / divider colour for reportlab
ACCENT_LINE: Final[str] = "#C4A882"     # warm sand

# Per-agent qualitative palette — 8 colours, one per company.
# Ordered to match the agent roster in configs (sellers first, then buyers).
AGENT_COLORS: Final[list[str]] = [
    "#B87D5B",  # Meridian Manufacturing    — warm terracotta
    "#7A9E82",  # Aldridge Industrial       — soft sage
    "#6E93B5",  # Crestline Components      — gentle slate blue
    "#C4827A",  # Vector Works              — dusty rose
    "#A89060",  # Halcyon Assembly          — warm sand / ochre
    "#5E8FA0",  # Pinnacle Goods            — muted teal
    "#B89878",  # Coastal Fabrication       — soft peach-brown
    "#907AA0",  # Northgate Systems         — soft mauve
]

AGENT_NAMES: Final[list[str]] = [
    "Meridian Manufacturing",
    "Aldridge Industrial",
    "Crestline Components",
    "Vector Works",
    "Halcyon Assembly",
    "Pinnacle Goods",
    "Coastal Fabrication",
    "Northgate Systems",
]

# Build a lookup: agent name → colour
AGENT_COLOR_MAP: Final[dict[str, str]] = dict(zip(AGENT_NAMES, AGENT_COLORS))

# Quality colour coding (for price charts etc.)
QUALITY_COLORS: Final[dict[str, str]] = {
    "Excellent": "#6E93B5",  # slate blue — premium
    "Poor": "#B87D5B",       # terracotta — economy
}

# Marker styles for scatter/line plots
QUALITY_MARKERS: Final[dict[str, str]] = {
    "Excellent": "o",
    "Poor": "s",
}


# ── Matplotlib style application ─────────────────────────────────────────────

def apply_matplotlib_style() -> None:
    """
    Apply the Sanctuary visual style to matplotlib globally.

    Call this once at the start of any script that generates plots.
    Uses rcParams rather than a .mplstyle file so the style travels
    with the package without needing file-system configuration.
    """
    mpl.rcParams.update(
        {
            # Figure
            "figure.facecolor": BACKGROUND,
            "figure.edgecolor": BACKGROUND,
            "figure.dpi": 150,
            # Axes
            "axes.facecolor": AXES_BACKGROUND,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            # Lines
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            # Ticks
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            # Legend
            "legend.framealpha": 0.85,
            "legend.facecolor": BACKGROUND,
            "legend.edgecolor": GRID_COLOR,
            "legend.fontsize": 8,
            # Colour cycle — set to our agent palette
            "axes.prop_cycle": mpl.cycler(color=AGENT_COLORS),
            # Save
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
            "savefig.bbox": "tight",
        }
    )


def agent_color(name: str) -> str:
    """Return the canonical colour for an agent, falling back to grey if unknown."""
    return AGENT_COLOR_MAP.get(name, "#999999")


def quality_color(quality: str) -> str:
    """Return the canonical colour for a quality tier."""
    return QUALITY_COLORS.get(quality, "#999999")


# ── Company roster ────────────────────────────────────────────────────────────
# Canonical ordering and role assignment — must match config YAML order.

SELLER_NAMES: Final[list[str]] = [
    "Meridian Manufacturing",
    "Aldridge Industrial",
    "Crestline Components",
    "Vector Works",
]

BUYER_NAMES: Final[list[str]] = [
    "Halcyon Assembly",
    "Pinnacle Goods",
    "Coastal Fabrication",
    "Northgate Systems",
]


# ── Report-specific palette ────────────────────────────────────────────────────

# Table header background
TABLE_HEADER_BG: Final[str] = "#EDE5D8"   # slightly deeper cream

# Alternating table row tint
TABLE_ROW_ALT: Final[str] = "#F5F0EA"     # very faint warm tint

# Page footer / caption text
CAPTION_COLOR: Final[str] = "#8C8070"     # warm mid-grey

# Misrepresentation highlight (for tables flagging deception)
MISREP_COLOR: Final[str] = "#C47A5A"      # muted burnt orange — visible but not garish


# ── ReportLab palette (for PDF generation) ───────────────────────────────────
# These are the same hues converted to (R, G, B) floats in [0, 1].

def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert #RRGGBB to a (r, g, b) tuple with values in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


REPORTLAB_BACKGROUND = hex_to_rgb(BACKGROUND)
REPORTLAB_ACCENT = hex_to_rgb(ACCENT_LINE)
REPORTLAB_TEXT_PRIMARY = hex_to_rgb(TEXT_PRIMARY)
REPORTLAB_TEXT_SECONDARY = hex_to_rgb(TEXT_SECONDARY)
REPORTLAB_CAPTION = hex_to_rgb(CAPTION_COLOR)
REPORTLAB_TABLE_HEADER = hex_to_rgb(TABLE_HEADER_BG)
REPORTLAB_TABLE_ALT = hex_to_rgb(TABLE_ROW_ALT)
REPORTLAB_MISREP = hex_to_rgb(MISREP_COLOR)
REPORTLAB_AGENT_COLORS_RGB = [hex_to_rgb(c) for c in AGENT_COLORS]
