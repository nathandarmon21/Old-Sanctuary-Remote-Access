"""Tests for the LLM-JSON-quirk normalizer.

Real-world LLM responses (Qwen on vLLM in particular) emit JSON with
``//`` line comments and trailing commas. The strict json.loads parser
rejects these; _normalize_llm_json strips them while preserving string
contents.
"""

from __future__ import annotations

import json

from sanctuary.agent import _normalize_llm_json, _parse_tactical_actions


class TestNormalizer:
    def test_strips_line_comments(self):
        s = '{"a": 1,  // a comment\n"b": 2}'
        cleaned = _normalize_llm_json(s)
        assert json.loads(cleaned) == {"a": 1, "b": 2}

    def test_strips_block_comments(self):
        s = '{"a": 1, /* note */ "b": 2}'
        assert json.loads(_normalize_llm_json(s)) == {"a": 1, "b": 2}

    def test_strips_trailing_commas(self):
        s = '{"a": [1, 2, 3,]}'
        assert json.loads(_normalize_llm_json(s)) == {"a": [1, 2, 3]}

    def test_preserves_url_in_string(self):
        """// inside string literals must NOT be stripped."""
        s = '{"url": "https://example.com/path"}'
        assert json.loads(_normalize_llm_json(s)) == {"url": "https://example.com/path"}

    def test_preserves_block_comment_marker_in_string(self):
        s = '{"text": "/* not a comment */ stays"}'
        assert json.loads(_normalize_llm_json(s)) == {"text": "/* not a comment */ stays"}

    def test_preserves_escaped_quote_in_string(self):
        s = r'{"text": "he said \"hi\" // not a comment"}'
        assert json.loads(_normalize_llm_json(s)) == {"text": 'he said "hi" // not a comment'}


class TestRealWorldVllmResponses:
    """Snippets from the actual vLLM pilot failures (job 9086783)."""

    def test_buyer_actions_with_inline_comments(self):
        # From d11_vllm_pilot v2 failure, slightly trimmed.
        text = '''### Tactical Decisions

#### Actions

<actions>
{
  "messages": [],
  "buyer_offers": [],
  "accept_offers": [
    "210f4101-5f04-4897-b6e5-a5765d1b968a",  // 8x Excellent at $45.00/unit
    "7ec2f421-8fc4-4ccd-8989-84975a69cf90"   // 8x Excellent at $45.00/unit
  ],
  "decline_offers": [
    "a24fd857-c332-43d0-97cc-4184cf803e5b"  // 2x Excellent at $50.00/unit
  ],
  "produce_final_goods": 0
}
</actions>
'''
        actions = _parse_tactical_actions(text, agent_role="buyer")
        # Critically: parse_error must be empty (clean parse, not failure).
        assert not actions.parse_error
        assert len(actions.accept_offers) == 2
        assert "210f4101-5f04-4897-b6e5-a5765d1b968a" in actions.accept_offers
        assert "a24fd857-c332-43d0-97cc-4184cf803e5b" in actions.decline_offers

    def test_seller_actions_with_markdown_fence_and_comments(self):
        text = '''### Plan
We will sell at $45.

```json
<actions>
{
  "messages": [],
  "offers": [
    {"to": "Halcyon Assembly", "qty": 1, "claimed_quality": "Excellent", "price_per_unit": 45.0}  // discounted
  ],
  "accept_offers": [],
  "decline_offers": [],
  "produce_excellent": 1,
  "produce_poor": 0,
  "build_factory": false
}
</actions>
```
'''
        actions = _parse_tactical_actions(text, agent_role="seller")
        assert not actions.parse_error
        assert len(actions.seller_offers) == 1
        assert actions.seller_offers[0].claimed_quality == "Excellent"
        assert actions.produce_excellent == 1
