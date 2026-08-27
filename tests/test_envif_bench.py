import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from envif_bench import aggregate, evaluate_item, family_of, verify_constraint  # noqa: E402


class TestEnvIFBench(unittest.TestCase):
    def test_numerical_load_requires_unit(self):
        cons = {"code": "numerical.value", "type": "numerical", "args": {"value": 2500.0, "unit": "kg/d"}}
        bad = verify_constraint("负荷 2500 t/d。", cons)
        good = verify_constraint("负荷 2500.00 kg/d。", cons)
        self.assertFalse(bad["passed"])
        self.assertTrue(good["passed"])

    def test_family_multi(self):
        cons = [
            {"code": "format.table", "type": "format", "args": {}},
            {"code": "numerical.value", "type": "numerical", "args": {"value": 1}},
        ]
        self.assertEqual(family_of(cons), "multi")

    def test_oracle_item_prompt_pass(self):
        item = {
            "id": "x",
            "task": "environmental_calculation",
            "constraint_family": "numerical",
            "query": "水量 7000 m3/d",
            "auto_verifiable": True,
            "constraints": [
                {"code": "numerical.value", "type": "numerical", "args": {"value": 80.95, "unit": "%"}},
                {"code": "numerical.formula", "type": "numerical", "args": {}},
            ],
        }
        resp = "去除率=(210-40)/210×100%=80.95%。"
        ev = evaluate_item(item, resp)
        self.assertTrue(ev["prompt_passed"], ev)

    def test_aggregate_keys(self):
        rows = [evaluate_item({
            "id": "a", "task": "format_constraint", "constraint_family": "format",
            "auto_verifiable": True, "query": "",
            "constraints": [{"code": "format.insufficient_prefix", "type": "format", "args": {}}],
        }, "信息不足。缺少标准。")]
        summ = aggregate(rows)
        self.assertEqual(summ["prompt_accuracy"], 1.0)
        self.assertIn("format", summ["by_family"])


if __name__ == "__main__":
    unittest.main()
