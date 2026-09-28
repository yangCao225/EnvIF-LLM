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

    def test_o2_formula_rejects_wrong_equation(self):
        query = "望海热电实测 SO2=260 mg/m³，烟气含氧 8.5%。请按基准氧 6% 折算。"
        cons = {"code": "numerical.formula", "type": "numerical", "args": {}}
        gold = verify_constraint(
            "公式 C'=C×(21−O2,s)/(21−O2,m)=260×(21−6)/(21−8.5)=312.00 mg/m³。",
            cons,
            query,
        )
        numeric = verify_constraint(
            "代入 C'=260×(21-6)/(21-8.5)=312.00 mg/m³。",
            cons,
            query,
        )
        inverted = verify_constraint(
            "C'=260×(21−8.5)/(21−6)=312.00 mg/m³。",
            cons,
            query,
        )
        fake = verify_constraint(
            "C_in=(10.74×Qg+C_in×O2_in)/(Qg+0.296×O2_in)。",
            cons,
            query,
        )
        self.assertTrue(gold["passed"], gold)
        self.assertTrue(numeric["passed"], numeric)
        self.assertFalse(inverted["passed"], inverted)
        self.assertFalse(fake["passed"], fake)

    def test_gas_formula_requires_24_e6(self):
        query = "临海热电厂烟气量 9300 m³/h，进口 SO2 820 mg/m³，出口 SO2 65 mg/m³。请计算去除率和每日去除负荷。"
        cons = {"code": "numerical.formula", "type": "numerical", "args": {}}
        ok = verify_constraint(
            "每日去除负荷=Qg×(C_in-C_out)×24×10^{-6}=168.52 kg/d。",
            cons,
            query,
        )
        ww = verify_constraint(
            "负荷=Q×(C_in-C_out)×0.001=7.02 kg/d。",
            cons,
            query,
        )
        self.assertTrue(ok["passed"], ok)
        self.assertFalse(ww["passed"], ww)

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
