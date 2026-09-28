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

    def test_chosen_hrt_includes_times_24(self):
        query = "二沉池容积 V=2000 m³，水量 Q=8000 m³/d，请计算水力停留时间 HRT。"
        gold = {"params": {"V": 2000, "Q": 8000}, "hrt_h": 6.0}
        chosen = chosen_response(query, gold, "wastewater_calculation")
        self.assertIn("×24", chosen)
        self.assertIn("6.00", chosen)
        self.assertIn("Q=8000", chosen)

    def test_chosen_load_uses_Q_not_V_for_flow(self):
        query = "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。"
        gold = {
            "params": {"Q": 10000, "Cin": 300, "Cout": 50},
            "removal_rate_pct": 83.33,
            "load_kg_d": 2500.0,
            "require_two_decimals": True,
            "require_magnitude_check": True,
        }
        chosen = chosen_response(query, gold, "wastewater_calculation")
        self.assertIn("水量 Q=10000", chosen)
        self.assertIn("Q×(C_in-C_out)×0.001", chosen)
        self.assertNotIn("已知V=", chosen)

    def test_hard_missing_0001_ranks_below_chosen(self):
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
        negs = typed_negatives(query, gold, chosen, "wastewater_calculation")
        miss = [n for n in negs if "Q×(C_in-C_out)=" in n["text"] and "×0.001" not in n["text"]]
        self.assertTrue(miss, msg=negs)
        ev_c = evaluate_response(chosen, query=query, gold=gold)
        ev_r = evaluate_response(miss[0]["text"], query=query, gold=gold)
        self.assertGreater(tlr_score(ev_c), tlr_score(ev_r), msg=miss[0]["text"])
        self.assertFalse(ev_r["calculation"]["passed"])

    def test_gas_chosen_uses_qg_and_24e6(self):
        query = "燃煤锅炉烟气量Qg=6850 m³/h，进口SO2=115 mg/m³，出口SO2=20 mg/m³。请计算去除率和每日去除负荷。"
        gold = {
            "params": {"Qg": 6850, "Cin": 115, "Cout": 20},
            "removal_rate_pct": 82.61,
            "load_kg_d": 15.62,
            "require_two_decimals": True,
            "require_magnitude_check": True,
            "knowledge_constraints": {"forbid_fabricated_standard": True, "process_family": "so2"},
        }
        chosen = chosen_response(query, gold, "air_pollution")
        self.assertIn("Qg=6850", chosen)
        self.assertIn("×24×10^{-6}", chosen)
        self.assertNotIn("负荷=Q×C×0.001", chosen)
        self.assertNotIn("Q×(C_in-C_out)×0.001", chosen)
        ev_c = evaluate_response(chosen, query=query, gold=gold)
        self.assertTrue(ev_c["passed"], ev_c)
        negs = typed_negatives(query, gold, chosen, "air_pollution")
        miss = [n for n in negs if "Qg×(C_in-C_out)=" in n["text"] and "24×10" not in n["text"]]
        self.assertTrue(miss, msg=negs)
        ev_r = evaluate_response(miss[0]["text"], query=query, gold=gold)
        self.assertGreater(tlr_score(ev_c), tlr_score(ev_r), msg=miss[0]["text"])
        self.assertFalse(ev_r["calculation"]["passed"])

    def test_o2_chosen_ranks_above_raw_concentration(self):
        query = "望海热电实测SO2=260 mg/m³，烟气含氧8.5%。请按基准氧6%折算。"
        gold = {
            "params": {"C": 260, "O2m": 8.5, "O2s": 6.0},
            "o2_corrected": 312.0,
            "require_two_decimals": True,
            "knowledge_constraints": {"forbid_fabricated_standard": True},
        }
        chosen = chosen_response(query, gold, "air_calculation")
        self.assertIn("(21−O2,s)/(21−O2,m)", chosen.replace(" ", ""))
        self.assertIn("312.00", chosen)
        ev_c = evaluate_response(chosen, query=query, gold=gold)
        self.assertTrue(ev_c["passed"], ev_c)
        negs = typed_negatives(query, gold, chosen, "air_calculation")
        raw = [n for n in negs if "当作折算浓度" in n["text"]]
        self.assertTrue(raw, msg=negs)
        ev_r = evaluate_response(raw[0]["text"], query=query, gold=gold)
        self.assertGreater(tlr_score(ev_c), tlr_score(ev_r))


if __name__ == "__main__":
    unittest.main()
