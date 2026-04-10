"""
Allow running sanctuary as a module: python -m sanctuary.run

This file dispatches to the appropriate entry point based on
how the package is invoked.
"""

from __future__ import annotations

import sys

# Default: run Mode 1 (batch)
from sanctuary.run import main

main()
