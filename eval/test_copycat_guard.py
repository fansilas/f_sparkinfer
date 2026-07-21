"""Unit tests for copycat reference PR selection + func-layer FP guards."""

import json
import unittest

import copycat_guard as cg


class CopycatGuardTests(unittest.TestCase):
    def test_list_reference_prs_filters_drafts(self):
        payload = [
            {"number": 10, "author": {"login": "alice"}, "isDraft": False},
            {"number": 11, "author": {"login": "bob"}, "isDraft": True},
            {"number": 12, "author": {"login": "carol"}, "isDraft": False},
        ]

        def fake_gh(args):
            class R:
                stdout = json.dumps(payload)
                returncode = 0
            return R()

        old = cg.gh
        cg.gh = fake_gh
        try:
            out = cg.list_reference_prs("owner/repo", limit=50)
            self.assertEqual([p["number"] for p in out], [10, 12])
        finally:
            cg.gh = old

    def test_list_reference_prs_uses_open_state(self):
        seen = {}

        def fake_gh(args):
            seen["args"] = args
            class R:
                stdout = "[]"
                returncode = 0
            return R()

        old = cg.gh
        cg.gh = fake_gh
        try:
            cg.list_reference_prs("owner/repo")
            self.assertIn("--state", seen["args"])
            self.assertEqual(seen["args"][seen["args"].index("--state") + 1], "open")
        finally:
            cg.gh = old

    def test_tiny_device_helper_is_boilerplate(self):
        sig = "__device__ __forceinline__ void pfm_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {"
        body = "\n".join([
            "    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }",
            "    else {",
            "        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);",
            "        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);",
            "    }",
        ])
        self.assertTrue(cg.is_boilerplate_block(sig, body))

    def test_large_kernel_not_boilerplate(self):
        sig = "__global__ void pfm_group_tilemap_kernel(const int* counts, int* out) {"
        body = "\n".join([f"    int x{i} = counts[{i}] + out[{i}];" for i in range(40)])
        self.assertFalse(cg.is_boilerplate_block(sig, body))

    def test_block_already_on_main(self):
        cache = {"kernels/csrc/cuda/fused/prefill_moe.cu": "\n".join([
            "__device__ void pfm_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {",
            "    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }",
            "    else {",
            "        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);",
            "        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);",
            "    }",
            "}",
        ])}
        body = "\n".join([
            "    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }",
            "    else {",
            "        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);",
            "        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);",
            "    }",
        ])
        self.assertTrue(cg.block_already_on_main(
            "owner/repo", "kernels/csrc/cuda/fused/prefill_moe.cu", body, cache))

    def test_func_layer_skips_pr566_style_fp(self):
        """PR #566: 27% PR-level + 100% on tiny main helper must not warn."""
        sig = "__device__ __forceinline__ void pfm_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {"
        body = "\n".join([
            "    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }",
            "    else {",
            "        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);",
            "        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);",
            "    }",
        ])
        self.assertFalse(cg.func_layer_should_warn(0.27, 1.0, sig, body))

    def test_func_layer_warns_on_large_embedded_kernel(self):
        sig = "__global__ void stolen_moe_kernel(const int* a, int* b) {"
        body = "\n".join([f"    int v{i} = a[{i}] * b[{i}] + {i}; __syncthreads();" for i in range(50)])
        self.assertTrue(cg.func_layer_should_warn(0.25, 0.95, sig, body))


if __name__ == "__main__":
    unittest.main()
