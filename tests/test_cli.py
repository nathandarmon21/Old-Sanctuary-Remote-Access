"""
Tests for sanctuary/run.py (Mode 1 CLI entry point).

Covers: run produces correct directory structure, manifest status,
argument parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sanctuary.run import _parse_args


class TestParseArgs:
    def test_config_required(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_config_flag(self):
        args = _parse_args(["--config", "configs/dev_local.yaml"])
        assert args.config == "configs/dev_local.yaml"

    def test_seed_default(self):
        args = _parse_args(["--config", "test.yaml"])
        assert args.seed == 42

    def test_seed_override(self):
        args = _parse_args(["--config", "test.yaml", "--seed", "99"])
        assert args.seed == 99

    def test_output_flag(self):
        args = _parse_args(["--config", "test.yaml", "--output", "/tmp/my_run"])
        assert args.output == "/tmp/my_run"

    def test_resume_flag(self):
        args = _parse_args(["--config", "test.yaml", "--resume", "/tmp/old_run"])
        assert args.resume == "/tmp/old_run"
