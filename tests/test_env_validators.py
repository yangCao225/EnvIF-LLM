import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from env_validators import evaluate_response, pollution_load_kg_d, removal_rate  # noqa: E402


class TestCalculations(unittest.TestCase):
    def test_cod_example(self):
        self.assertAlmostEqual(removal_rate(300, 50), 83.3333, places=2)
        self.assertAlmostEqual(pollution_load_kg_d(10000, 250), 2500.0)

    def test_correct_response_passes_calc(self):
        query = "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。"
        response = (
            "已知条件：Q=10000 m³/d，Cin=300 mg/L，Cout=50 mg/L。\n"
            "去除率=(300-50)/300×100%=83.33%。\n"
            "每日去除负荷=10000×(300-50)×0.001=2500.00 kg/d。\n"
            "数量级合理性检查：日去除2.5 t COD，对于万吨级污水厂合理。"
        )
        gold = {
            "params": {"Q": 10000, "Cin": 300, "Cout": 50},
            "removal_rate_pct": 83.33,
            "load_kg_d": 2500.0,
            "require_two_decimals": True,
            "require_magnitude_check": True,
        }
        result = evaluate_response(response, query=query, gold=gold)
        self.assertTrue(result["calculation"]["passed"], result)
        self.assertTrue(result["knowledge"]["passed"], result)

    def test_unit_aware_load_rejects_wrong_mass_unit(self):
        gold = {"params": {"Q": 10000, "Cin": 300, "Cout": 50}, "removal_rate_pct": 83.33, "load_kg_d": 2500.0}
        bad = evaluate_response("去除率83.33%，每日去除负荷为 2500.00 t/d。", gold=gold)
        good = evaluate_response("去除率83.33%，每日去除负荷为 2500.00 kg/d。数量级合理。", gold=gold)
        self.assertFalse(bad["calculation"]["passed"], bad)
        self.assertIn("unit_error", bad["error_types"])
        self.assertTrue(good["calculation"]["passed"], good)

    def test_false_certainty_without_standard(self):
        q = "未提供执行标准，请判断出水是否达标。"
        bad = evaluate_response("根据经验已经达标。", query=q)
        good = evaluate_response("信息不足。未提供标准名称和年份，需核实现行标准。", query=q)
        self.assertFalse(bad["knowledge"]["passed"])
        self.assertIn("false_certainty", bad["error_types"])
        self.assertTrue(good["knowledge"]["passed"])

    def test_insufficient_prefix(self):
        gold = {"format_constraints": {"must_start_insufficient": True}, "knowledge_constraints": {"require_insufficient": True}}
        ok = evaluate_response("信息不足。缺少水量和排放标准年份。", query="数据不足", gold=gold)
        bad = evaluate_response("已经达标。", query="数据不足", gold=gold)
        self.assertTrue(ok["format"]["passed"])
        self.assertFalse(bad["format"]["passed"])

    def test_fabricated_standard(self):
        q = "请分析该出水是否超标，未提供执行标准。"
        bad = evaluate_response("按照GB 18918-2002一级A限值COD为50 mg/L，已经达标。", query=q)
        good = evaluate_response("未提供标准依据，需核实现行标准，不能判定是否超标。", query=q)
        self.assertFalse(bad["knowledge"]["passed"])
        self.assertTrue(good["knowledge"]["passed"])

    def test_seed_eval_funcs_compile(self):
        from env_seed_validators import SEED_EVAL_FUNCS
        from utils import compile_eval_func
        for name, code in SEED_EVAL_FUNCS.items():
            fn = compile_eval_func(code)
            self.assertIsNotNone(fn, name)
            self.assertIsInstance(fn("dummy"), bool)

    def test_removal_seed_func(self):
        from env_seed_validators import SEED_EVAL_FUNCS
        from utils import compile_eval_func
        fn = compile_eval_func(SEED_EVAL_FUNCS["去除率"])
        self.assertTrue(fn("进水300 mg/L，出水50 mg/L，去除率=(300-50)/300=83.33%"))
        self.assertFalse(fn("COD降低了"))

    def test_config_and_domain_registered(self):
        from config_loader import get
        self.assertEqual(get("data.domain"), "环境工程")
        self.assertAlmostEqual(float(get("training.dpo.learning_rate")), 5e-6)
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from extended_domains import EXTENDED_DOMAINS
        self.assertIn("环境工程", EXTENDED_DOMAINS)
        self.assertGreaterEqual(len(EXTENDED_DOMAINS["环境工程"]["seed_instructions"]), 30)
        self.assertEqual(
            EXTENDED_DOMAINS["污水处理"]["seed_instructions"],
            EXTENDED_DOMAINS["环境工程"]["seed_instructions"],
        )


if __name__ == "__main__":
    unittest.main()
