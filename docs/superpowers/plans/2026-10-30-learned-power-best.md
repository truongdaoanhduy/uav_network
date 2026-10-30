# Learned-power restoration and best-policy plan
Spec:docs/superpowers/specs/2026-10-30-learned-power-best-design.md
Inline execution authorized by explicit implementation request; keep existing feature worktree.

## Constraints and review focus
1. Never promote current live weights for an older asynchronous evaluation.
2. Validation bank alone selects; final-test/legacy-bank results cannot leak into selection.
3. Selection state survives fresh-directory BASE and PQL continuation without RNG draws; old checkpoints work.
4. Distinguish final training budget from earlier selected-policy provenance; cloud artifacts keep exact-resume state separate.
5. Reuse matched episode windows, source/science/precision labels and root stitching; sparse tails cannot win.
No file over500 source lines; preserve current dirty continuation record; no cloud launches.

## Task1: Evidence and learned-power baseline
- [ ] Fetch sanitized live run index and unsampled histories, compare common episode windows across all substantive runs; inventory best historical candidate artifacts.
- [ ] Exercise current-profile learned actions and generated guided-power ablation (RED before profile change).
- [ ] Change the three current profile modes/names/tags, baseline validation and default guided ablation name; preserve historical manifests.
- [ ] Run controlled-power/study/sharing/production-identity tests and commit.

## Task2: Best-policy selection and exact continuation
Interfaces:BestCheckpoint(config,output,state=None), consider(result,source), state_dict(), result(last_path). AsyncCheckpointEvaluator(selection_state=None) owns it. Optional training.best_checkpoint in fixed/PQL full state.
- [ ] Write RED behavioral tests for ranking, exact actor bytes, strict provenance/banks, resumed earlier winner, corruption and RNG.
- [ ] Implement atomic best selection/metadata and async promotion/upload, optional full-state serialization/restore.
- [ ] Route trainer result checkpoint to selected best and label fallback; preserve final learner resume state. Add truthful W&B best metadata/alias.
- [ ] Validate study against selected source step/hash and completed training budget.
- [ ] Verify real evaluator, BASE/PQL, exact-resume, tracking,study suites and commit.

## Task3: Review and handoff
- [ ] Report matched-window results, transmission/relay assumptions and available-vs-missing historical checkpoints with timestamps.
- [ ] Full pytest, compileall,diff check and credential-value scan.
- [ ] One fresh whole-change review; fix important findings with RED→GREEN.
- [ ] Commit evidence/current task record and push normal feature branch; verify remote SHA.
