# Exact run `18d756ef` — frozen training/validation/test/Warp PHY/MAC reference

**Reference-only files, not executable as part of M2.** These files were read from the verified source tar inside Kaggle notebook `therwed/notebook5051433b29`. Archive SHA256 `4f66fbb12ac2ff68675f2bb1590fb7066610ced96da6b264272ea5fe6b9e4aa2`. File SHA256s are in `manifest.json`. The current M2 implementations of these modules deliberately remain distinct.

This reference preserves the actual `training.py`, `one_session.py`, `deferred_gpu_evaluation.py`, hybrid replay implementation, and Warp vector/MAC/report code so the **E4096 U4 B8192 75/25, validation100 and final4000** workflow can be ported with explicit M2 semantics.

**Never import these copies into a live M2 run.** Old GPU PHY/MAC/report code uses old target identities and Bayesian belief, NOT M2 source-provenanced packet events, DS/SPRT and GCS `MAP_DELTA`. Port each mechanism only after Native-Warp mission/packet parity tests.
