"""Opt-in, read-only loss/gradient/parameter forensic timeline for parallel UAV RL.

The complete layerwise gradient path is already in learner-layer-snapshots;
this adds the matching Actor head/encoder parameter and feature statistics.
No new backward passes, sampling, optimizers, or RNG consumption. Each
snapshot uses a fixed prefix of the *actual replay minibatch* rather than
inventing a synthetic state. Always correlate by global_step + actor_updates.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


class RootCauseAudit:
    def __init__(self, output, algorithm, *, enabled=None, envs, learning_starts,
                 normal_interval, max_observations=32):
        self.enabled = (os.environ.get('UAV_ROOT_CAUSE_AUDIT', 'false').lower()
                        in ('1','true','yes')) if enabled is None else bool(enabled)
        self.algorithm = str(algorithm)
        self.envs = int(envs)
        self.learning_starts = int(learning_starts)
        self.normal_interval = max(self.envs, int(normal_interval))
        self.max_observations = int(max_observations)
        self.path = Path(output) / 'root-cause-timeline.jsonl'
        self.last_step = None
        self.records = 0
        self.param_snapshot = None
        self.reference_observations = None
        self.reference_encoder_state = None
        self.reference_head_weight = None
        self.reference_head_bias = None
        self.reference_shared = None
        self.reference_num_uavs = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

    def due(self, step):
        """Every vector step in first ~0.5M transitions, then decimate."""
        if not self.enabled or step < self.learning_starts:
            return False
        after = step - self.learning_starts
        interval = (self.envs if after <= 393216 else
                    self.envs * 4 if after <= 1966080 else self.normal_interval)
        return after % interval == 0

    def all_updates_due(self, step):
        """Probe EVERY optimizer update at warmup onset + the next 16 collections.

        Original collapse arose within 10 MATD3 Actor updates. This captures
        the true onset instead of a single last minibatch every collection.
        """
        return (self.enabled and step >= self.learning_starts and
                (step-self.learning_starts)%self.envs == 0 and
                step-self.learning_starts <= self.envs*16)

    def action_due(self, step):
        """Warmup + dense early action/mission samples, then sparse monitoring."""
        if not self.enabled or step <= 0 or step % self.envs:
            return False
        if step < self.learning_starts:
            return step == self.envs or step % (self.envs*4) == 0
        after=step-self.learning_starts
        return (after <= self.envs*16 or
                (after <= self.envs*512 and after%(self.envs*4)==0))

    @staticmethod
    def _value(value):
        if isinstance(value, Real):
            v = float(value)
            return v if math.isfinite(v) else ('NaN' if math.isnan(v) else 'Inf' if v > 0 else '-Inf')
        if torch.is_tensor(value) and value.numel() == 1:
            return RootCauseAudit._value(value.item())
        return None

    @staticmethod
    def _flatten_observations(observations, *, shared, num_uavs):
        if shared:
            b,n = observations.shape[:2]
            ids = torch.arange(n,device=observations.device).expand(b,n)
            return torch.cat((observations,F.one_hot(ids,num_uavs).to(observations.dtype)),-1).reshape(b*n,-1)
        return observations[:,0]

    @torch.no_grad()
    def remember_reference(self, actor, observations, *, shared, num_uavs):
        """Freeze real replay observations AND the pre-update Actor for attribution.

        Called exactly once before the first learner update. Never calls
        .sample() or an optimizer, so model/RNG/training state is untouched.
        """
        if not self.enabled or self.reference_observations is not None:
            return
        self.reference_observations = observations[:self.max_observations].detach().clone()
        self.reference_shared = bool(shared)
        self.reference_num_uavs = int(num_uavs)
        self.reference_encoder_state = {k:v.detach().clone() for k,v in actor.encoder.state_dict().items()}
        head = actor.continuous if self.algorithm == 'matd3' else actor.mean
        self.reference_head_weight = head.weight.detach().clone()
        self.reference_head_bias = head.bias.detach().clone()

    @torch.no_grad()
    def reference_attribution(self, actor):
        """Factorial 2x2 head/encoder ablation on an identical frozen state.

        Quantifies where output drift originates but does NOT infer why Q
        gradients/optimizer updates originally favored that drift.
        """
        if self.reference_observations is None:
            return {}
        obs = self._flatten_observations(self.reference_observations,
            shared=self.reference_shared,num_uavs=self.reference_num_uavs)
        from torch.func import functional_call
        old_h = F.relu(functional_call(actor.encoder,self.reference_encoder_state,(obs,)))
        new_h = F.relu(actor.encoder(obs))
        old_w,old_b = self.reference_head_weight,self.reference_head_bias
        head = actor.continuous if self.algorithm == 'matd3' else actor.mean
        z00 = F.linear(old_h,old_w,old_b)
        z10 = F.linear(old_h,head.weight,head.bias)
        z01 = F.linear(new_h,old_w,old_b)
        z11 = F.linear(new_h,head.weight,head.bias)
        l2 = lambda t: t.square().mean().sqrt().item()
        return {
            'reference/pretrain_abs_mean':z00.abs().mean().item(),
            'reference/head_only_abs_mean':z10.abs().mean().item(),
            'reference/encoder_only_abs_mean':z01.abs().mean().item(),
            'reference/current_abs_mean':z11.abs().mean().item(),
            'reference/head_only_delta_l2':l2(z10-z00),
            'reference/encoder_only_delta_l2':l2(z01-z00),
            'reference/total_delta_l2':l2(z11-z00),
            'reference/interaction_delta_l2':l2(z11-z10-z01+z00),
            'reference/current_motion_tanh_derivative':(1-z11[:,:3].tanh().square()).mean().item(),
        }

    @torch.no_grad()
    def actor_snapshot(self, actor, observations, *, shared, num_uavs):
        """Direct attribution z = W*h + bias on a representative replay subset.

        Shared-actor identity matches the exact learner _actor_input mapping.
        For MASAC the deterministic mean is observed, never sample/r-sample.
        """
        obs = observations[:self.max_observations]
        if shared:
            b, n = obs.shape[:2]
            ids = torch.arange(n, device=obs.device).expand(b,n)
            features = torch.cat((obs, F.one_hot(ids,num_uavs).to(obs.dtype)), -1)
            features = features.reshape(b*n,-1)
        else:
            features = obs[:,0]
        head = actor.continuous if self.algorithm == 'matd3' else actor.mean
        hidden = F.relu(actor.encoder(features))
        raw_linear = F.linear(hidden, head.weight)  # isolate W*h, not bias
        raw = raw_linear + head.bias
        motion = raw[:,:3]
        tanh_derivative = 1.0-torch.tanh(motion).square()
        finite = torch.isfinite(raw)
        out = {
            'head/actor_batchnorm_enabled': int(any(isinstance(m, nn.BatchNorm1d) for m in actor.encoder.modules())),
            'head/batchnorm_running_mean_abs_mean': (
                sum(m.running_mean.abs().mean().item() for m in actor.encoder.modules()
                    if isinstance(m, nn.BatchNorm1d)) /
                max(1,sum(isinstance(m, nn.BatchNorm1d) for m in actor.encoder.modules()))),
            'head/batchnorm_running_var_mean': (
                sum(m.running_var.mean().item() for m in actor.encoder.modules()
                    if isinstance(m, nn.BatchNorm1d)) /
                max(1,sum(isinstance(m, nn.BatchNorm1d) for m in actor.encoder.modules()))),
            'head/continuous_weight_l2':head.weight.norm().item(),
            'head/continuous_weight_abs_max':head.weight.abs().max().item(),
            'head/continuous_bias_l2':head.bias.norm().item(),
            'head/weighted_feature_abs_mean':raw_linear.abs().mean().item(),
            'head/raw_pretanh_abs_mean':raw.abs().mean().item(),
            'head/raw_pretanh_abs_max':raw.abs().max().item(),
            'head/motion_pretanh_abs_mean':motion.abs().mean().item(),
            'head/motion_tanh_derivative_xyz_mean':tanh_derivative.mean().item(),
            'head/motion_tanh_derivative_lt_1e3_fraction':(tanh_derivative<1e-3).float().mean().item(),
            'head/motion_tanh_output_gt_0_99_fraction':(motion.tanh().abs()>.99).float().mean().item(),
            'head/hidden_abs_mean':hidden.abs().mean().item(),
            'head/hidden_zero_fraction':(hidden==0).float().mean().item(),
            'head/finite_fraction':finite.float().mean().item(),
            'head/destination_weight_l2':actor.destination.weight.norm().item(),
            'head/destination_bias_l2':actor.destination.bias.norm().item(),
        }
        for i,axis in enumerate('xyz'):
            out[f'head/pre_tanh_{axis}_abs_mean']=motion[:,i].abs().mean().item()
            out[f'head/pre_tanh_{axis}_signed_mean']=motion[:,i].mean().item()
        for i,linear in enumerate(m for m in actor.encoder.modules() if isinstance(m,nn.Linear)):
            out[f'encoder/linear_{i}_weight_l2']=linear.weight.norm().item()
            out[f'encoder/linear_{i}_bias_l2']=linear.bias.norm().item()
        for i,ln in enumerate(m for m in actor.encoder.modules() if isinstance(m,nn.LayerNorm)):
            out[f'encoder/layernorm_{i}_gamma_l2']=ln.weight.norm().item()
            out[f'encoder/layernorm_{i}_beta_l2']=ln.bias.norm().item()
            out[f'encoder/layernorm_{i}_gamma_max']=ln.weight.abs().max().item()
        if self.algorithm=='masac':
            log_std=actor.log_std(hidden)
            out['head/log_std_weight_l2']=actor.log_std.weight.norm().item()
            out['head/log_std_abs_mean']=log_std.abs().mean().item()
            out['head/log_std_clamp_low_fraction']=(log_std<=-20).float().mean().item()
            out['head/log_std_clamp_high_fraction']=(log_std>=2).float().mean().item()
        return out

    @torch.no_grad()
    def capture(self, step, actor, observations, optimizer, *, actor_updates,
                critic_updates, num_uavs, shared, replay=None, update_index=None,
                pre_update_head=None):
        if not self.enabled:
            return False
        identity=(int(step), None if update_index is None else int(update_index))
        if self.last_step == identity:
            return False
        head = self.actor_snapshot(actor, observations, shared=shared, num_uavs=num_uavs)
        reference = self.reference_attribution(actor)
        metrics = {}
        accepted=('gradprobe/','layerprobe/','masac_probe/','actor_', 'critic_', 'q_',
                  'td_error_', 'continuous_entropy','discrete_entropy','alpha_',
                  'actor_output_penalty','policy_updated')
        for k,v in optimizer.items():
            if isinstance(k,str) and k.startswith(accepted):
                val=self._value(v)
                if val is not None:metrics[k]=val
        record={'schema':1, 'audit_scope':'sampled_update_not_causal_proof',
                'step':int(step),'actor_updates':int(actor_updates),
                'critic_updates':int(critic_updates),'algorithm':self.algorithm,
                'optimizer_update_index':update_index,
                'before_update_head':pre_update_head,
                'observation_scope':'first_32_rows_of_replay_minibatch',
                'optimizer':metrics, 'head':head,'reference':reference,
                'replay':{k:self._value(v) for k,v in (replay or {}).items()
                          if isinstance(k,str) and self._value(v) is not None}}
        with self.path.open('a') as out:
            out.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
        self.last_step=identity
        self.records+=1
        return True
