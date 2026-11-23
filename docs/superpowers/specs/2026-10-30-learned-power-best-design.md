# Restore learned power and select validated best policy
Date:2026-10-30. Base:e78a9da26688ca48415da438d5c3aa43b2d9e605.
User explicitly requests checking27–29Oct MASAC/MATD3 including all sharing/replay variants at matched episodes, returning to learned-power actions, and saving the best policy instead of always choosing the last one.

## Authorized behavior
Keep current6-UAV/1000m/20T15O/1200-step profiles and sharing/replay settings. Return all three current production YAMLs to power_control=learned with honest names/tags. Retain guided+3dB as the separate link_budget ablation; historical35/25 YAMLs and existing manifests remain unchanged.

Use existing periodic held-out CPU validation bank, not training rolling reward or final test bank. Rank candidates lexicographically by success_rate,mean_delivery_rate_percent,mean_search_rate_percent,mean_return; equal scores keep the earlier candidate. All-zero success still distinguishes partial delivery/search. Only valid finite results bound to exact immutable source bytes/config/step/seed bank may win. Legacy validation on the final-test bank cannot select best.

Promote the evaluated actor snapshot, not the learner at result-arrival time. Atomically write best.pt and best metadata. It is evaluation-only; resume_latest.pt remains the exact last learner/optimizer/replay/RNG/environment state. Result checkpoint and final W&B model artifact/CPU visualization use best.pt when a validated candidate exists. If no valid candidate completed, keep an explicitly labeled unvalidated last-policy fallback; never fabricate best. Preserve the existing bounded shutdown, so no hidden long final evaluator is added to time-capped cloud training.

Carry selected snapshot bytes and metadata in optional full-checkpoint training state so a continuation in a fresh directory does not lose an earlier winner. Validate scientific/algorithm/precision/training-seed/validation-bank identity while permitting operational output path changes. Old checkpoints without selection state remain valid. Do not perturb parent RNG to restore this actor-only state.

Study summary validates the selected policy's own training step (possibly earlier than completed training), while separately requiring the full training budget. Emit both final training budget and selected-policy episode/step. Bind selected bytes and mode/precision, including BASE/PQL swap rejection. W&B final artifact may retain final/recovery aliases for workflow compatibility and add best alias only for validated-best policies.

## Evidence study
Query W&B read-only live; include earlier roots active around27–29Oct. Separate train/eval/diagnostic, FP16/FP32, sharing flags, E/B, scientific/source settings. Download unsampled per-episode history, stitch root/continuations without duplicate episodes, compare identical episode windows and expose actual global steps. Do not rank unmatched whole-run means, tiny partial tail blocks or seed44 as universal evidence. Inventory available checkpoint versions before claiming historical best recoverable; historical train-peak is not validated best.

## Verification and limits
Behavioral RED→GREEN tests for exact old-best preservation after later worse policy, success/delivery tie-breaks, invalid/wrong-bank rejection, parent-RNG stability, fresh-directory resume winner carry, legacy state, real CPU evaluator and BASE/PQL selected output; unchanged resume state and provenance rejection.
Run full pytest, compileall,diff/credential checks, independent final review and normal working-branch commit/push per AGENTS. Preserve initial dirty CURRENT_TASK suffix. No production cloud launch or cancellation; no3D/sensing changes; no claim of universal convergence or CPU/GPU quality effects.
