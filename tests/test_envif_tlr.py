import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from env_validators import evaluate_response  # noqa: E402
from envif_tlr import ABLATION_SETS, chosen_response, tlr_score, typed_negatives  # noqa: E402


class TestEnvIFTLR(unittest.TestCase):
    def test_full_layers_rank_chosen_above_typed_negatives(self):
        query = "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。"
        gold = {
            "params": {"Q": 10000, "Cin": 300, "Cout": 50},
            "removal_rate_pct": 83.33,
            "load_kg_d": 2500.0,
            "require_two_decimals": True,
            "require_magnitude_check": True,
            "knowledge_constraints": {"forbid_fabricated_standard": True},
        }
        chosen = chosen_response(query, gold, "wastewater_calculation")
        ev_c = evaluate_response(chosen, query=query, gold=gold)
        self.assertTrue(ev_c["passed"], ev_c)
        for neg in typed_negatives(query, gold, chosen):
            ev_r = evaluate_response(neg["text"], query=query, gold=gold)
            self.assertGreater(
                tlr_score(ev_c),
                tlr_score(ev_r),
                msg=neg["error_type"],
            )

    def test_calc_layer_alone_cannot_catch_fabricated_standard(self):
        query = "未提供执行标准，请判断是否达标。"
        gold = {"knowledge_constraints": {"forbid_fabricated_standard": True}}
        good = "信息不足。缺少标准名称与年份，需核实现行标准。"
        bad = "按照 GB 18918-2002 一级A限值 COD 为 50 mg/L，已经达标。"
        ev_g = evaluate_response(good, query=query, gold=gold)
        ev_b = evaluate_response(bad, query=query, gold=gold)
        self.assertGreater(tlr_score(ev_g, ("knowledge",)), tlr_score(ev_b, ("knowledge",)))
        # 计算层在无 gold 数字时会 skip，单靠计算层排不出知识错误
        self.assertEqual(tlr_score(ev_g, ("calculation",)), tlr_score(ev_b, ("calculation",)))
        self.assertIn(("format", "calculation", "knowledge"), ABLATION_SETS)


if __name__ == "__main__":
    unittest.main()
