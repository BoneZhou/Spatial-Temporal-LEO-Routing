import unittest
from .revision_burst_analysis import recovery

class RecoveryProtocolTests(unittest.TestCase):
    def test_first_qualifying_start_not_confirmation(self):
        q = [10.0]*500 + [30.0]*250 + [10.0]*750
        r = recovery(q, 500, 750)
        self.assertAlmostEqual(r["recovery_time_s"], 23*.02)
        self.assertEqual(r["q_pre"], 10)

    def test_requires_full_stable_window(self):
        q = [10.0]*500 + [30.0]*250 + [10.0]*30
        self.assertEqual(recovery(q,500,750)["status"], "right_censored")

    def test_no_minimum_plus_two_threshold(self):
        q = [1.0]*500 + [3.0]*250 + [1.2]*750
        self.assertIsNone(recovery(q,500,750)["recovery_time_s"])

    def test_uses_only_last_one_second_of_baseline(self):
        q = [100.0]*450 + [10.0]*50 + [30.0]*250 + [10.0]*750
        self.assertEqual(recovery(q,500,750)["q_pre"], 10)

if __name__ == "__main__":
    unittest.main()
