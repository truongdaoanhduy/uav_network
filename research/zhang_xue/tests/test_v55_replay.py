from __future__ import annotations
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from zhang_xue.learners.state_derived_replay import StateDerivedReplayBuffer,estimate_replay_bytes

class StateDerivedReplayContract(unittest.TestCase):
    def test_preserves_contract_and_reconstructs_exact_concat_state(self):
        buff=StateDerivedReplayBuffer(capacity=8,num_agents=6,obs_dim=514,state_dim=3084,
                                      cont_dim=2,disc_dim=7,device=torch.device('cpu'))
        o=torch.randn((2,6,514),dtype=torch.float32)
        m=torch.ones((2,6,7),dtype=torch.uint8)
        buff.add_batch(obs=o,state=o.flatten(1),next_obs=o*2,next_state=(o*2).flatten(1),
                       act_cont=torch.zeros((2,6,2)),act_disc_onehot=torch.zeros((2,6,7)),
                       rewards=torch.tensor([1.,2.]),dones=torch.tensor([0.,1.]),mask=m,next_mask=m)
        b=buff.sample(20)
        self.assertEqual(b['state'].shape,(20,3084))
        self.assertTrue(torch.equal(b['state'],b['obs'].reshape(20,-1)))
        self.assertTrue(torch.equal(b['next_state'],b['next_obs'].reshape(20,-1)))
        self.assertEqual(buff.size,2)
        self.assertEqual(buff.vram_bytes(),buff.estimated_bytes)
        self.assertEqual(buff.vram_bytes(),estimate_replay_bytes(8,6,514,2,7,False))

    def test_rejects_non_concat_state(self):
        with self.assertRaises(ValueError):
            StateDerivedReplayBuffer(capacity=8,num_agents=6,obs_dim=514,state_dim=100,
                                      cont_dim=2,disc_dim=7,device=torch.device('cpu'))

if __name__=='__main__': unittest.main()
