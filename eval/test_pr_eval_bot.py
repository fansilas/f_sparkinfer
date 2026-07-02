#!/usr/bin/env python3
"""Unit tests for PR bot rendering/policy metadata.

Run from the repo root:
  python3 eval/test_pr_eval_bot.py
"""
import unittest

import pr_eval_bot as bot


class PrEvalBotPolicyTest(unittest.TestCase):
    def test_regression_labels_block_automerge(self):
        self.assertIn("regression-128", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-512", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-4k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-16k", bot.AUTOMERGE_BLOCK_LABELS)

    def test_mixed_win_render_keeps_eval_label_and_shows_regression(self):
        res = {
            "label": "S",
            "pass": True,
            "tps": 205.0,
            "frontier_tps": 195.0,
            "delta_tps": 10.0,
            "pct_over_frontier": 5.1,
            "top1": 0.97,
            "kl": 0.02,
            "eval_mode": "longctx",
            "score_context": 4096,
            "best_context_label": "4k-context",
            "ctx_128_tps": 470.0,
            "guard_128_baseline": 481.0,
            "guard_128_pass": False,
            "ctx_512_tps": 406.0,
            "guard_512_baseline": 405.0,
            "guard_512_pass": True,
            "ctx_4096_tps": 205.0,
            "guard_4k_baseline": 195.0,
            "guard_4k_pass": True,
            "ctx_16384_tps": 266.0,
            "guard_16k_baseline": 265.0,
            "guard_16k_pass": True,
            "regression_labels": ["regression-128"],
        }
        body = bot.render(res, "abc1234")
        self.assertIn("`eval:S`", body)
        self.assertIn("4096 ctx · 4k-context", body)
        self.assertIn("regression-128", body)
        self.assertNotIn("Auto-closing", body)

    def test_auto_close_reject_render_explains_regression_only_case(self):
        res = {
            "label": "REJECT",
            "pass": False,
            "auto_close": True,
            "reason": "512-context decode no-regression gate failed",
            "tps": 401.0,
            "frontier_tps": 405.0,
            "delta_tps": -4.0,
            "pct_over_frontier": -1.0,
            "top1": 0.97,
            "kl": 0.02,
            "eval_mode": "longctx",
            "score_context": 512,
            "best_context_label": "512-context",
            "ctx_512_tps": 401.0,
            "guard_512_baseline": 405.0,
            "guard_512_pass": False,
            "regression_labels": ["regression-512"],
        }
        body = bot.render(res, "def5678")
        self.assertIn("`eval:REJECT`", body)
        self.assertIn("regression-512", body)
        self.assertIn("Auto-closing this PR", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
