"""No-regression gates: any failed guard rejects — gains elsewhere do not excuse it.

Mirrors evaluate.sh: reject when EVAL_MODE != short and ALL_GUARDS_PASS != true.
PR #562 scored eval:L with +13.6% @64k pp while 128k pp collapsed (287 vs 17020)
because a verified mid-ctx gain previously skipped the REJECT path.

CB mixed-load TTFT (latency, lower-is-better) uses the same tol: fail when
ttft > frontier / tol. Single-seq pp gains must not excuse a serving-path regression.
"""
import unittest


def should_reject_for_guards(all_guards_pass: bool, eval_mode: str = "longctx") -> bool:
    if eval_mode == "short":
        return False
    return not all_guards_pass


def cb_ttft_guard_pass(ttft: float, frontier: float, tol: float = 0.98) -> bool:
    """Latency no-regression: allow up to ~(1/tol - 1) worse TTFT vs main."""
    if ttft <= 0 or frontier <= 0:
        return True  # skip when unmeasured (same as missing decode baselines)
    return ttft <= frontier / tol


class GuardRejectTests(unittest.TestCase):
    def test_pr562_128k_pp_fail_with_64k_gain_rejects(self):
        # Score path had HAS_VERIFIED_CONTEXT_GAIN from 64k pp +13.6%; 128k pp failed.
        all_guards_pass = False  # guard_128k_pp_pass == false
        self.assertTrue(should_reject_for_guards(all_guards_pass))

    def test_all_guards_pass_does_not_reject(self):
        self.assertFalse(should_reject_for_guards(True))

    def test_short_mode_skips_guard_reject(self):
        self.assertFalse(should_reject_for_guards(False, eval_mode="short"))

    def test_cb_ttft_regression_fails_gate(self):
        # PR #591-shaped: 0.505s vs main 0.463s (−9.1%) must fail at 2% tol.
        self.assertFalse(cb_ttft_guard_pass(0.505, 0.463, tol=0.98))

    def test_cb_ttft_within_tol_passes(self):
        # 1% worse latency is within 2% tol.
        self.assertTrue(cb_ttft_guard_pass(0.4676, 0.463, tol=0.98))

    def test_cb_ttft_improvement_passes(self):
        # PR #584-shaped latest: 0.468s vs main 0.542s.
        self.assertTrue(cb_ttft_guard_pass(0.468, 0.542, tol=0.98))

    def test_cb_ttft_unmeasured_skips(self):
        self.assertTrue(cb_ttft_guard_pass(0.0, 0.542))
        self.assertTrue(cb_ttft_guard_pass(0.468, 0.0))


if __name__ == "__main__":
    unittest.main()
