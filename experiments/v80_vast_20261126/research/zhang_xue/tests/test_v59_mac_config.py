"""CSMA timings and PHY settings must be adjustable from validated config.

These settings control the CPU reference only; existing Warp's compile-time
timings require an explicit new parity/cuda gate before use in training.
"""
import inspect
import sys
from pathlib import Path
import unittest

SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.communication.common import CsmaProtocolConfig
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.communication.scheduler import _NetworkSchedulerMixin
from zhang_xue.communication.channel import _NetworkChannelMixin
from zhang_xue.communication.delivery import _NetworkDeliveryMixin


class ConfigDrivenMAC(unittest.TestCase):
    def test_protocol_default_matches_historical_timing(self):
        p=CsmaProtocolConfig()
        self.assertEqual((p.difs_us,p.sifs_us,p.slot_us,p.cw_min,p.max_retries,p.ack_timeout_us),
                         (50,10,20,31,5,180))

    def test_network_direct_protocol_override(self):
        p=CsmaProtocolConfig(difs_us=70,slot_us=25,max_retries=3,ack_timeout_us=250)
        net=SevenRadioCsmaCaNetwork(protocol=p)
        self.assertIs(net.protocol,p)

    def test_env_profile_cli_to_real_cpu_radio(self):
        cfg=load_scenario(overrides=(
            "mac_difs_us=70","mac_sifs_us=12","mac_slot_us=25",
            "mac_cw_min=15","mac_max_retries=3",
            "mac_ack_timeout_us=190","mac_cca_threshold_dbm=-80",
            "mac_noise_power_w=8.0e-11","mac_queue_capacity=32",
        ))
        env=ZhangXue7NodePacketEnv(cfg,seed=44)
        self.assertEqual(env.net.protocol.difs_us,70)
        self.assertEqual(env.net.protocol.slot_us,25)
        self.assertEqual(env.net.protocol.max_retries,3)
        self.assertEqual(env.net.queue_cap,32)
        self.assertAlmostEqual(env.net.noise_power_w,8e-11)
        self.assertAlmostEqual(env.net.cca_th_w,10**((-80-30)/10))

    def test_invalid_timings_fail_fast(self):
        for bad in [
            {"difs_us":0}, {"slot_us":-1}, {"cw_min":0},
            {"max_retries":0}, {"ack_timeout_us":120},
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):CsmaProtocolConfig(**bad)

    def test_scheduler_and_rx_paths_use_protocol_settings(self):
        sched=inspect.getsource(_NetworkSchedulerMixin)
        chan=inspect.getsource(_NetworkChannelMixin)
        delivery=inspect.getsource(_NetworkDeliveryMixin)
        for value in ("difs_us","sifs_us","slot_us","cw_min","max_retries"):
            self.assertIn("self.protocol."+value,sched+delivery)
        self.assertIn("self.protocol.ack_timeout_us",chan)
        self.assertIn("self.protocol.sifs_us",delivery)


if __name__=="__main__": unittest.main()
