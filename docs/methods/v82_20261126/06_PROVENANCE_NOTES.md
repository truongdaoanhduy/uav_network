# Code and result provenance



The case15 MATD3 warmup-only diagnostic began before the log-interval instrumentation edit. Case26 repeats it on clean code commit e81bab955cde37c87addeca606906040fa4e93fd. Keep both; they are not independent seeds.

Case17 MATD3 900-step run began 53 seconds after the final runner edit, while files were still uncommitted; it uses the final code content later committed as e81bab9. Case18 and final23 smoke runs have clean e81bab9 provenance.

Original V81 cases03/04 use exactly the attached source_v81, not the edited runner.



Code file digests:

scripts/gpu/run_v81_real_rollout.py f5f285ee697aa88adce2ae494d30ffa8ef14fdb9b1b5b6c098bb88f8c9efce96

scripts/gpu/rollout_v82_contracts.py 8554708dc4ad6c9c007025a234171c9f49baea29bba16628e9509728c782d4c1

uav_network/algorithms/configuration/schema.py aa0ebeaaf4cdece45e97993fce6935ef567d1235aa22da591a6ad00d4116225e

uav_network/algorithms/core/models.py 0a1f443026a665930acf907f1dda1bf583098cbab6ae88ccd26a95f1f26b0ee9

research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py b08963f87e41f21e8a616bdf5a594a49cd9ebe16302ad8c082ed3aad963c72b8
