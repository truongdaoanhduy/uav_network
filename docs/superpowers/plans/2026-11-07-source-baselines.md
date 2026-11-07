# Independent MASAC and MATD3 baseline implementation plan

> For agentic workers use superpowers:executing-plans to implement the verified steps.

The user requests source-faithful baselines and an audit of four completed runs.
The shared scalar team critic is a valid different objective. A new independent
plain-MLP profile selects per-agent critics and the original SAC temperature
objective, while historical configurations retain their serialized behavior.

- [x] Reproduce and fix the Kaiming notebook label with a generated-header test.
- [x] Test the original alpha-weighted temperature gradient at nonunit alpha.
- [x] Add the explicit temperature-loss setting, validated defaults, and old
  checkpoint/config identity compatibility.
- [x] Add a matched independent actor/critic profile with plain MLPs, default
  MATD3 initialization, NOCHUNK, and declared hybrid/mask interfaces.
- [x] Verify per-agent critic targets, BASE training/checkpoint reload, source
  contracts and historical checkpoint compatibility.
- [x] Record matched W&B evaluation, raw reward components, paper differences,
  exact source pins and the limits of one-seed initialization results.

Primary temperature source is rail-berkeley/softlearning at
13cf187cc93d90f7c217ea2845067491c3c65464, sac.py 245-264.
Per-agent hybrid objectives are Fu et al., CoRL/PMLR270, Eqs. 4, 8-11.
MATD3 source is Ackermann et al. arXiv:1910.01465v2 and its released trainer.

Main pair was prioritized: 30 new common layouts at 18.432M plus a sampled-MASAC diagnostic. Full evidence is linked from the main-pair methods audit.
