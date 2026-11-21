"""Sparse, read-only per-layer actor/critic gradient/activation instrumentation.

Runs ONLY when collect_metrics=True on an actual update. Parameter gradients are
observed before optimizer gradient clipping (autograd parameter hooks) and
after clipping (existing parameter.grad). Forward/output gradient statistics
use up to 256 elements to avoid dense activation copies or extra forward /
backward passes. The user-visible metric names mark this sampling explicitly.

This observer does not change actor/critic weights, RNG, learning losses, or
execution path; it should never be enabled for every optimizer step of a large
benchmark without an intentional resource budget.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager

import torch
from torch import nn


class LayerwiseGradientProbe(AbstractContextManager):
    def __init__(self, modules, kind, *, enabled=False, sample_elements=256,
                 near_zero_threshold=1e-12):
        if kind not in {"actor", "critic"}:
            raise ValueError("Expected actor or critic layer probe")
        self.modules = modules
        self.kind = kind
        self.enabled = enabled
        self.sample_elements = int(sample_elements)
        self.threshold = float(near_zero_threshold)
        self.handles = []
        self.parameters = defaultdict(list)
        self.forward_stats = defaultdict(list)
        self.backward_stats = defaultdict(list)
        self.preclip_stats = defaultdict(list)
        self.expected = defaultdict(int)
        self.hooked = defaultdict(int)

    @staticmethod
    def _linear_layer_names(module, kind):
        if kind == "actor":
            for index, layer in enumerate(m for m in module.encoder.modules()
                                          if isinstance(m, nn.Linear)):
                yield f"encoder_{index}", layer
            for index, layer in enumerate(m for m in module.encoder.modules()
                                          if isinstance(m, nn.LayerNorm)):
                yield f"encoder_layernorm_{index}", layer
            yield "motion_head", module.continuous
            yield "destination_head", module.destination
        else:
            for tower_name in ("q1", "q2"):
                tower = getattr(module, tower_name)
                for index, layer in enumerate(m for m in tower.modules()
                                              if isinstance(m, nn.Linear)):
                    yield f"{tower_name}_layer_{index}", layer

    def __enter__(self):
        if not self.enabled:
            return self
        for module in self.modules:
            for label, linear in self._linear_layer_names(module, self.kind):
                self.expected[label] += 1
                self.parameters[label].append(linear.weight)
                self.handles.append(linear.weight.register_hook(
                    lambda grad, label=label: self._param_hook(label, grad)))
                self.handles.append(linear.register_forward_hook(
                    lambda _module, _inputs, output, label=label:
                    self._forward_hook(label, output)))
            if self.kind == "actor":
                named_modules=[("encoder", module.encoder)]
            else:
                named_modules=[("q1",module.q1),("q2",module.q2)]
            for block_name, seq in named_modules:
                for index, relu in enumerate(m for m in seq.modules()
                                             if isinstance(m, nn.ReLU)):
                    label=f"{block_name}_relu_{index}"
                    self.handles.append(relu.register_forward_hook(
                        lambda _module,_inputs,output,label=label:
                        self._relu_hook(label, output)))
        return self

    def _relu_hook(self,label,output):
        if not output.requires_grad:
            return
        with torch.no_grad():
            sample=output.detach().flatten()[:self.sample_elements]
            self.forward_stats[label].append(
                {"zero_fraction":(sample==0).float().mean()})

    def _forward_hook(self,label,output):
        if not output.requires_grad:
            return
        with torch.no_grad():
            sample=output.detach().flatten()[:self.sample_elements]
            self.forward_stats[label].append({
                "abs_mean":sample.abs().mean(),
                "near_zero_fraction":(sample.abs()<self.threshold).float().mean(),
                "nonfinite_fraction":(~torch.isfinite(sample)).float().mean(),
            })
        self.handles.append(output.register_hook(
            lambda grad,label=label:self._output_backward_hook(label,grad)))

    def _output_backward_hook(self,label,grad):
        with torch.no_grad():
            sample=grad.detach().flatten()[:self.sample_elements]
            self.backward_stats[label].append({
                "abs_mean":sample.abs().mean(),
                "near_zero_fraction":(sample.abs()<self.threshold).float().mean(),
                "exact_zero_fraction":(sample==0).float().mean(),
                "nonfinite_fraction":(~torch.isfinite(sample)).float().mean(),
            })
        return None

    def _param_hook(self,label,grad):
        with torch.no_grad():
            g=grad.detach()
            # Full weight tensor; sampled only during diagnostic updates.
            self.preclip_stats[label].append({
                "norm":torch.linalg.vector_norm(g),
                "abs_mean":g.abs().mean(),
                "near_zero_fraction":(g.abs()<self.threshold).float().mean(),
                "exact_zero_fraction":(g==0).float().mean(),
                "nonfinite_fraction":(~torch.isfinite(g)).float().mean(),
            })
            self.hooked[label]+=1
        return None

    @torch.no_grad()
    def metrics(self):
        if not self.enabled:
            return {}
        out={}
        prefix=f"layerprobe/{self.kind}"
        for label in sorted(self.expected):
            key=f"{prefix}/{label}"
            out[key+"/parameter_count"]=self.expected[label]
            out[key+"/parameter_hook_count"]=self.hooked[label]
            if self.preclip_stats[label]:
                values=torch.stack([x["norm"] for x in self.preclip_stats[label]])
                out[key+"/preclip_norm_min_across_modules"]=values.min()
                out[key+"/preclip_norm_max_across_modules"]=values.max()
            out[key+"/missing_gradient_fraction"]=(self.expected[label]-self.hooked[label])/max(1,self.expected[label])
            groups=(
                ("preclip",self.preclip_stats),
                ("forward_sample",self.forward_stats),
                ("backward_sample",self.backward_stats),
            )
            for marker,records in groups:
                if not records[label]:
                    continue
                metrics=records[label][0].keys()
                for metric in metrics:
                    out[f"{key}/{marker}_{metric}"]=torch.stack([
                        record[metric] for record in records[label]]).mean()
            active=[p.grad.detach() for p in self.parameters[label] if p.grad is not None]
            if active:
                post_norms=torch.stack([torch.linalg.vector_norm(g) for g in active])
                out[key+"/postclip_grad_norm"]=post_norms.mean()
                out[key+"/postclip_norm_min_across_modules"]=post_norms.min()
                out[key+"/postclip_norm_max_across_modules"]=post_norms.max()
                out[key+"/postclip_exact_zero_fraction"]=torch.stack([
                    (g==0).float().mean() for g in active]).mean()
                out[key+"/postclip_near_zero_fraction"]=torch.stack([
                    (g.abs()<self.threshold).float().mean() for g in active]).mean()
        for label,records in self.forward_stats.items():
            if "relu_" in label and records:
                out[f"{prefix}/{label}/forward_sample_zero_fraction"]=torch.stack(
                    [record["zero_fraction"] for record in records]).mean()
        out[prefix+"/sample_elements_per_activation"]=self.sample_elements
        out[prefix+"/num_linear_layers_monitored"]=sum(self.expected.values())
        out[prefix+"/expected_parameter_gradient_hooks"]=sum(self.expected.values())
        out[prefix+"/received_parameter_gradient_hooks"]=sum(self.hooked.values())
        return out

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
