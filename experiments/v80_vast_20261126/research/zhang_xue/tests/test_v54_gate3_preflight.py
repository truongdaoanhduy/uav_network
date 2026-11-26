"""Gate 3 rejects an incomplete GPU port despite 80 correctly counted schema arrays."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zhang_xue.gpu.kernels.scaffold import audit_gate3_warp_parity_contract

class Gate3Readiness(unittest.TestCase):
    def test_18_incomplete_schema_fails_closed(self):
        s = audit_gate3_warp_parity_contract()
        self.assertFalse(s['gate3_ready'])
        self.assertEqual(s['declared_bytes_per_env'],490001)
        self.assertIn('ack_wm_ts',s['minimum_missing_fields_bytes_per_env'])
        self.assertIn('gcs_last_rx_us',s['minimum_missing_fields_bytes_per_env'])
        self.assertGreater(s['minimum_known_bytes_per_env'],490001)
