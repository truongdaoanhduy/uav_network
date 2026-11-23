# MAP800 posterior-only TX0.8W: full Kaggle notebook in therwed/notebook5051433b29 format

**Date**: 2026-11-23. Based on commit `81870800f41ee6837dc2e042c568f65335f4899b`, branched from `experiment/map800-tx08-learned-hf-deep-forensics-20261123`. This is a **separate scientific task/MDP** and new Actor observation-version 11. Previous historical and TX0.4/0.8 LR4/5W runs remain unchanged. No actual Kaggle GPU training has been launched by notebook creation.

## Exact reference format

The user's requested public Kaggle notebook was pulled using authenticated read-only `kaggle kernels pull therwed/notebook5051433b29 -m` and inspected: it has **exactly three cells**, namely Markdown introduction, editable code preflight/configuration, and one self-contained base85-embedded Python source bootstrap which runs Train → two-GPU Validation → Best → held-out Test. This new notebook preserves that format, not a separate non-production 64-step pilot.

**Importable notebook**: `notebooks/MATD3-NOORACLE-POSTERIORONLY-MAP800-TX08-E4096-U4-B8192-H256x256-ORTHO-COMPILE-VAL100-TEST1000-T4X2-FULLFLOW-20261123.ipynb`.

**User operation**: Import `.ipynb` at Kaggle; choose GPU T4×2, enable Internet, grant notebook access to Kaggle Secrets `WANDB_API_KEY` and `HF_TOKEN`, then Save Version → Save & Run All. `NOTEBOOK_REF` auto-reads `KAGGLE_KERNEL_REF` if available; falls back to an explicitly marked placeholder for provenance, not a real actual Kaggle URL. Run ID is unique and emitted in notebook output. No GitHub fetch or credential content in the file. The source archive is fully embedded and checked SHA-256 on Kaggle before installation. It refuses CPU fallback.

## Bayesian confirmation — exact functional change

- **Before** (old branch, posterior+LR): a cell was confirmed if sensor positive `yes`, Bayes posterior>=0.99 AND effective positive likelihood ratio>=4. At 150m with Pd=.7 and Pf=.3 full-cell LR≈2.33, even high posterior was rejected. Historical `18d756ef` used posterior>=.99 plus **altitude<=50m**; it did not have an independent LR>=4 gate.
- **Now**: `confirmation_mode=posterior_only`, `confirmation_min_likelihood_ratio=0.0` (**disabled**), `confirmation_threshold=0.99`. The only target-confirmation evidence gate is a positive camera sample and posterior>=0.99, at any **active** camera altitude z>0. Camera FOV/Pd/Pf, partial coverage q, occlusion, Bayes update and no-oracle cell-ID reporting remain active. No ground truth is revealed to Actor or used to veto a false candidate. The investigation potential in this mode is proximity/evidence only, not an implicit descent-to-LR or descent-to-50m shaping reward. It remains a fresh MDP requiring training from scratch.
- **Checkpoint observation-version 11** rejects old v8/9/10 Actor checkpoints; historical LR4 v10 behavior and historical fine-altitude rules are kept unchanged for their respective configs.
- Positive false detection risk is real. In a **16-world×1500-step Warp CPU stress with handcrafted exploratory actions**, the new posterior-only scenario produced 407 false confirmed cell hypotheses, versus 195 for the previous LR-gated MAP800 TX0.8 stress with the same actions; confirmed true deliveries increased from 399 to 493. These are NOT trained MATD3 results, nor independent evidence of improved policy performance. Monitor false confirmations, false reports delivered, report slots140, PDR and energy.

## Core science & GPU protocol

- 6 UAVs; 35 ground targets; 25 obstacles; map800×800m, 40×40 Bayesian grid with 20m cells; target confirmation posterior>=.99, no LR gate, altitude0–150m and 90° FoV.
- MATD3 with Actor BN + Critic plain, Actor/Critic H256×256, orthogonal initialization, `torch.compile` Inductor. E4096, U4, B8192, hybrid replay1M 75%CPU/25%GPU, seed44, max1500 steps per episode, target20,000 complete episodes. Baseline algorithm initialization is **random fresh**, not historic checkpoint.
- Learned radio TX continuous log mapping [0.001,0.8]W, separate silence/5 peer relay/direct GCS destination. ACK/retries/interference and PHY losses, battery shared with flight. Reward RF Joule reference fixed at0.4W to prevent penalty dilution from changed Pmax.
- Actor snapshots at initial step0, every100 vector steps and final; all snapshots kept. `one_session` with explicit **validate_all_snapshots=True** disables hidden candidate shortlist (the older code by default might select at most32).
- GPU Validation: **100 fixed held-out seeds/checkpoint** and up to81 policies×100=8100 active worlds within8192 physical Warp environments per GPU. `contiguous_adaptive` assignment preserves CP1-81 on GPU0 and CP82-162 on GPU1 when >=163 policies, or splits a short prefix approximately evenly between both GPUs so GPU1 is not idle for typical ~70 snapshots.
- Best Actor chosen on the fixed Validation100 scores; **Final Test 1,000 held-out episodes** =500/GPU for the same Best Actor, no sharing Test seeds or selection bias. The one-session W&B supervisor resumes exactly the training run ID for Eval/Test and reports completion only if every checkpoint and all Test episodes finish.
- Kaggle T4×2 user must select hardware and authorize Secrets. **CUDA/VRAM for E4096 train & E8192 eval, 20k convergence and 12h job completion are unverified**. If Kaggle session time expires or OOM occurs, notebook fails explicitly rather than inventing fullflow success. The requested E8192 physical pool is an upper-world configuration, not proof a T4 can fit that memory.

## Deep W&B/HF forensic data

- Online W&B: vector-step reward and health, gradient/Q/tanh diagnostics, mission and network metrics, TX power histogram edges and counts, per-episode RF joules and propulsion joules, routing fractions, packet loss reasons, collisions, PDR, SINR, queue state, false confirmations and false report deliveries, Validated checkpoint mean returns and held-out Test result.
- HF public diagnostic dataset: immutable compressed forensic archive shards, interval ~300s, integrity manifest with SHA-256 and public readback; `forensic-transitions.jsonl` selected worlds0,1 every16 vector steps (dense around learner warmup), full belief and pre/post position, RF action and true TX/RX energy, exact per-packet power and destination; `forensic-minibatches.jsonl` replay Q/grad/tanh; `root-cause-timeline.jsonl`, `training-step-reward/progress/optimizer.jsonl`, `episodes.jsonl`, run provenance and error journals. Captures representative environments, **NOT every transition from all 4096 worlds**.
- `UAV_ROOT_CAUSE_AUDIT=true`, `UAV_FORENSIC_ENABLED=true`, `UAV_WANDB_PLOTS_ONLY=false`, `UAV_DIAGNOSTIC_STORE=huggingface_public`, `UAV_HF_VERIFY_PUBLIC_READBACK=true` are injected by the notebook before training. Login credentials come exclusively from granted Kaggle Secrets. No uploads occurred in local CPU tests.

## Verification on current Ubuntu CPU

1. New mode real Warp confirmation succeeds at150m when high posterior and positive yes; old LR-gated mode still rejects at150m; historical mode still 50m. Legacy configuration/Actor version unchanged.
2. Explicit false-positive test demonstrates a false cell may be confirmed at150m (no ground-truth oracle veto), correctly reported by diagnostics.
3. Posterior-only investigation potential is independent of altitude for the same belief/proximity; no concealed LR descent reward.
4. Real compiled MATD3 CPU train from exactly the embedded self-contained Kaggle source tar -> version11 `final.pt` -> held-out CPU Eval2 episodes passed. No downloading source at runtime.
5. Real compiled MATD3 CPU train with W&B/HF **fake network transports** passed, including non-empty deep journals with actual TX, RX and replay minibatches and verified reconstruction of HF archives. No public uploaded output claimed.
6. GPU batch layout and zero-shortlist opt-in tested, existing contiguous_batches/strided modes retained; historical unit tests still pass.
7. Selected regression groups: **118 passed +14 subtests** and **93 passed (7 warnings)**, plus new posterior-only HF real integration 1 pass. Total **212 passing tests +14 subtests**; NOT exhaustive repository test suite.
8. 16 CPU Warp full-horizon worlds ×1500 steps complete with zero internal Warp errors, altitude up to150m, max report hypotheses68/140. This is manually scripted action stress, NOT trained-policy evidence.

## Artifacts

- `configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml`
- `scripts/build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123.py`
- `notebooks/MATD3-NOORACLE-POSTERIORONLY-MAP800-TX08-E4096-U4-B8192-H256x256-ORTHO-COMPILE-VAL100-TEST1000-T4X2-FULLFLOW-20261123.ipynb`
- `tests/test_nooracle_map800_posterior_only_tx08_20261123.py`
- `tests/test_nooracle_posterioronly_gpu_eval_fullflow_layout_20261123.py`
- `tests/test_nooracle_map800_posterioronly_kaggle_reference_format_20261123.py`
- `tests/test_nooracle_map800_posterioronly_bundled_cpu_fullflow_20261123.py`
- `tests/test_nooracle_map800_posterioronly_hf_real_log_20261123.py`
