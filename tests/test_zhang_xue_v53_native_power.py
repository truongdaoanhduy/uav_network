"""Requested native TX power must never silently become another experiment."""
import os
from pathlib import Path
import unittest

from uav_network.communication.classic_v1_simpy import (
    ClassicV1SimPyNetwork,
    DEFAULT_V1_ROOT,
)
from uav_network.communication.zhang_xue_ref.ds_belief import BeliefTable


class NativePowerContractTests(unittest.TestCase):
    def test_env_forwards_unsupported_power_to_native_backend(self):
        from uav_network.environment.zhang_xue_coop import (
            ZhangXueConfig,
            ZhangXueCooperativeEnv,
        )
        with self.assertRaisesRegex(ValueError, "tx_power_w"):
            ZhangXueCooperativeEnv(
                ZhangXueConfig(tx_power_w=0.02),
                network_backend="classic_simpy_v1",
            )

    def test_unsupported_power_fails_before_loading_upstream(self):
        # Replacing a rejected setting by the native 0.5-W default would
        # silently change the experiment. No pinned checkout is needed to
        # validate that the requested power is supported.
        for power in (0.02, 0.1, 2.0, 0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(power=power):
                try:
                    ClassicV1SimPyNetwork(
                        {0: (0.0, 0.0), 1: (1.0, 0.0)},
                        BeliefTable(2, 400),
                        v1_root="/nonexistent/uav-classic-power-test",
                        tx_power_w=power,
                    )
                except ValueError as error:
                    self.assertIn("tx_power_w", str(error))
                except Exception as error:
                    self.fail(f"Power must fail closed before native loading: {error}")
                else:
                    self.fail("Unsupported native power was accepted")

    def test_rejected_override_cannot_change_a_live_native_network(self):
        root = Path(os.environ.get("UAV_CLASSIC_V1_ROOT", DEFAULT_V1_ROOT))
        if not (root / "entities/drone.py").is_file():
            self.skipTest("Pinned classic source unavailable")
        positions = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        first = ClassicV1SimPyNetwork(positions, BeliefTable(2, 400), v1_root=root)
        with self.assertRaisesRegex(ValueError, "tx_power_w"):
            ClassicV1SimPyNetwork(
                positions, BeliefTable(2, 400), v1_root=root, tx_power_w=2.0,
            )
        # Exercise the real native PHY module's shared power, not just the
        # first network's cached diagnostic scalar.
        import utils.config as native_config
        self.assertEqual(first.tx_power_w, 0.5)
        self.assertEqual(native_config.TRANSMITTING_POWER, 0.5)
        second = ClassicV1SimPyNetwork(
            positions, BeliefTable(2, 400), v1_root=root, tx_power_w=0.5,
        )
        self.assertEqual(second.tx_power_w, 0.5)
        self.assertEqual(native_config.TRANSMITTING_POWER, 0.5)


if __name__ == "__main__":
    unittest.main()
