# Five MATD3 Kaggle UI notebooks

User explicitly requests five BASE notebooks, no MASAC export or submission.
Use the existing 35-target/25-obstacle,6-UAV,1km,1500-step scenario,seed44.

| Case | Sharing actor+critic | Hidden actor+critic | Updates/collection | Actor LayerNorm |
|---|---|---|---|---|
|01|shared|512,512|8|true|
|02|shared|512,512|16|true|
|03|independent|512,512|16|true|
|04|independent|512,512|16|false|
|05|shared|256,256|8|true|

LayerNorm means the existing actor_layer_norm option. Critics remain plain
MLPs; do not introduce critic normalization or change algorithm equations.
All other runtime/scientific settings are the existing scaled profile:
default MATD3 initialization, output_normalization=false, policy_delay2,
NOCHUNK,vectorize_critics=false,FP32,compile/AMP/fused off,batch8192,
4096env,replay1M split75%CPU/25%GPU,warmup131072,20000 episodes.
Keep reward,CBF,energy,sensing/network and validation seeds/schedule.

Build from one clean committed source archive. Use the existing kernel
bootstrap and Kaggle Secrets; no credentials embedded, no Git clone.
Each file has exact case labels/config overrides and a fresh W&B identity
per launch, seed editing propagates correctly, no warm-start/auto-continue.
Export distinct filenames preserving all prior notebooks. Training-only
with existing W&B tracking/HF validation-best upload; no visualization.
Fresh Kaggle sessions,Private,T4x2,Internet,real NOTEBOOK_REF required.

Approach selected by implementation judgment: one reusable five-case
builder over the existing scaled profile, explicit resolved overrides.
Do not duplicate whole scenario YAMLs or hand-edit encoded source payloads.
No GPU training or API submission; CPU/schema tests are not convergence
or full-batch GPU memory evidence.
