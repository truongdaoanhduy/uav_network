# 2026-11-25 — v4.9 source-backed Warp CSMA busy-carrier freeze component

Branch experiment/zhang-xue-v49-native-csma-freeze-20261125 from v4.8 d12895b. Actual pinned upstream UavNetSim-v1 CsmaCa.mac_send SimPy generator with external occupied channel validated against new Warp kernel for source-exact DIFS restart, frozen remaining backoff, 20us carrier poll and attempts1-5 CW doubling. Nine directly measured first-TX scenarios all match source at zero microsecond error; seed45 fifth retry sends DATA at absolute 49756us after relay E2E ACK occupies 48530..49020us. E4096 world and mixed attempt-window tests also PASS. Evidence docs/superpowers/evidence/zhang_xue_v49_native_csma_freeze_cpu_20261125.json; docs docs/superpowers/NETWORK_V49_SOURCE_PINNED_WARP_CSMA_FREEZE_20261125.md.

Gate5 NOT PASSED: helper models ONE sender with EXTERNAL busy intervals, not integrated continuous multi-UAV PHY/MAC; real CUDA and training/eval/test NOT launched.

---

# 2026-11-25 — v4.8 Warp application envelope + six-UAV five-hop DATA and E2E ACK

Branch experiment/zhang-xue-v48-hop-envelope-six-uav-20261125, from v4.7 commit f302638. Actual Warp queue-side immutable app envelope (original origin, ultimate destination, six-bit visited, hop count), loop/hop budget checks and frozen source evidence through every relay. Physically received 90B HELLO+DV discover 5-hop static folded chain, source camera DATA transfers UAV0→1→2→3→4→5, then real 90B app E2E ACK returns UAV5→4→3→2→1→0. E4096 controlled CPU: 4096 original DATA, 3072 final DATA/3072 matching E2E ACK source watermarks, 1024 original DATA TTL expirations while relay2 permanently in blackout; 17,408 DATA/15,360 ACK hop receipts, zero loops/duplicate ACK or premature DS. Evidence docs/superpowers/evidence/zhang_xue_v48_4096_fivehop_e2e_blackout_cpu_20261125.json and detailed report docs/superpowers/NETWORK_V48_SIX_UAV_FIVE_HOP_ENVELOPE_20261125.md.

Full Gate5 still BLOCKED, no production actor, dynamic multi-hop under concurrent load, source-exact SimPy MAC, CUDA, training/eval/test. Keep main untouched.

---

# 2026-11-25 — v4.7 Warp path-vector 3-node loop prevention

Branch experiment/zhang-xue-v47-pathvector-loop-guard-20261125 from v4.6 commit307e785. TDD RED reproduces 3-UAV loop 0→2→1→0 after link0↔3 expires, even after v4.6 poisoned reverse. New Warp six-bit path masks per destination and per physically received DV advertisement reject routes containing recipient anywhere, preserve source/peer/destination paths, and retain HELLO/DV TTL reception barriers. E4096 CPU and valid 2hop tests pass. See docs/superpowers/NETWORK_V47_DISTRIBUTED_PATH_MASK_20261125.md.

Full Gate5 still blocked: only controlled DV metadata packet, no native source-exact CSMA, fully fused six-agent dynamic multihop control, actual CUDA, production reward/energy/actor. Training NOT launched.

---

# 2026-11-25 -- v4.6 poisoned reverse prevents stale distributed route loops

Branch experiment/zhang-xue-v46-dv-poisoned-reverse-20261125, parent v4.5 commit1020f2d. New TDD RED regression proved Warp DV could create loop 1→0→1 after 1↔2 HELLO expiry because 0 re-advertised a stale route to 1 that depended on 1. Implemented source-neighbor split horizon with poisoned reverse on physically received DV only (unreachable INF when sender next-hop equals DV receiver). 3 new tests including E4096 and old static multi-hop tests pass. See docs/superpowers/NETWORK_V46_DV_POISONED_REVERSE_20261125.md.

Still not full UavNetSim/CUDA Gate5, no MATD3/MASAC training launched. Keep main untouched.

---

# 2026-11-25 — v4.5 sampled real 1km moving network, truth-only diagnostic reward

Branch experiment/zhang-xue-v45-sampled-moving-network-20261125, parent v4.4 SHA a180130. New pilot uses existing 6 UAV/10 target/2 no-fly/2 blackout Zhang-Xue scenario sampler, Warp safe movement and 200-step horizon, per-radio MAC HELLO discovery refreshed every four steps, actor-local source evidence and receiver-origin DS, persistent queues and true reverse E2E APP ACK watermark, with correct accounting for multiple DATA arriving at one receiver or multiple ACKs to one source. E128x200 seed44: mean coverage79.207%, true confirmation964/1280, false36, DATA admitted66416, received57287, ACK attempted57287 with 1341 queue-full rejections and 50723 valid E2E watermark updates, 16212 radio blackout UAV-steps, no unsafe final positions/receiver-leak. E4096x4 seed53 also ran. See docs/superpowers/NETWORK_V45_SAMPLED_1KM_MOVING_MAC_DS_ACK_20261125.md and evidence JSON.

Gate5 still BLOCKED: full production reward/energy/actor missing, six-UAV learned multi-hop incomplete, source-exact SimPy MAC event parity and CUDA not verified. Training still NOT LAUNCHED.

---

# 2026-11-25 — v4.4 real E2E ACK watermarks and six-sender contended CPU smoke

Branch experiment/zhang-xue-v44-e2e-watermarks-concurrent-20261125, derived from v4.3 6c2ada5. Warp source-peer persistent 400-cell ACK watermarks advance ONLY from physically received matching application E2E ACK, never MAC ACK. Stale/duplicate ACK ignored; select dirty cells up to62 per snapshot; resend after application timeout. Six UAVs learn a static ring from MAC HELLO, then concurrently generate MAP-origin evidence packets, contend using Warp CCA/SINR/retry, and exchange DATA and APP E2E ACK; receiver DS updates only on DATA. CPU E256x200 seed44: 126091 DATA generated, 119974 physically received, 119215 APP ACK/watermark updates, 6385 timed-out sends released for resend, zero remote belief leakage/blackout sends; E4096x4: 45622 DATA, 42186 received, 40922 APP ACK. Evidence at docs/superpowers/NETWORK_V44_WARP_E2E_WATERMARKS_SIX_UAV_CONTENDED_20261125.md.

Gate5 still NOT PASSED: ring not dynamic real multi-hop, no full source-exact SimPy MAC parity, full actor/motion/reward/energy or CUDA. Training remains blocked.

---

# 2026-11-25 — v4.3 private account2 CUDA preflight

Branch experiment/zhang-xue-v43-kaggle-cuda-preflight-20261125, parent v4.2 SHA ee58650. Fixed source allowlist ZIP embedded in private Kaggle script from immutable SHA, no API token in source, CUDA E16/E1024/E4096 smoke and CPU parity, T4x2 target. Test covers generator. Kaggle CLI v1.7.4 fails KGAT; isolated venv CLI v2.2.5 authenticates account2 and confirmed GPU quota 29.53h remaining. See docs/superpowers/NETWORK_V43_PRIVATE_KAGGLE_CUDA_PREFLIGHT_20261125.md. **Not training or Gate5 completion.**

---

# 2026-11-25 — v4.2 Warp receiver evidence / immutable queued payload / no-oracle routing

Branch experiment/zhang-xue-v42-warp-evidence-routing-20261125, from v4.1 commit 555f2a8. Tests 82/82 focused regressions passed including pinned SimPy. E4096 Warp CPU controlled 3-node 2-hop experiment: real 90B MAC HELLO and DV packets establish 0→1→2 path without physical oracle; DATA from source camera counts uses queued immutable provenance payload; recipient-local origin MAX-plus-SUM DS updated only on DATA success; 3072/4096 DATA final deliveries, 3072 two-hop E2E ACK, 1024 DATA TTL expirations; queues empty, no blackout sends, no evidence leak. See docs/superpowers/NETWORK_V42_WARP_RECEIVER_PAYLOAD_DISTRIBUTED_ROUTING_20261125.md and evidence JSON.

**Gate 5 is NOT PASSED. No training.** Missing real concurrent multiagent load, fully fused GPU queue/MAC/routing, exact native CSMA freeze/backoff, full 200-step actor/reward scenarios and CUDA run. Keep all previous branches unchanged.

---

# 2026-11-25 — v4.1 Warp persistent queue and app E2E ACK preflight

Branch `experiment/zhang-xue-v41-persistent-warp-queue-20261125`, parent v4.0 `169348f`. Implemented source-owned packet **metadata** queue on actual Warp tensors (6 UAVs/E worlds; bounded capacity; FIFO; duplicate/full detection; one outstanding transmission per source; exact integer-microsecond TTL; blackout retention/recovery; stale-completion rejection; epoch reset). One-hop MAC + app E2E ACK synthetic 7-step 4096-world CPU smoke: 4096 DATA generated, 3072 DATA and 3072 app E2E ACK received, 1024 expired, zero blackout sends, queue empty. 12 new focused tests passed. See `docs/superpowers/NETWORK_V41_PERSISTENT_QUEUE_AND_APP_ACK_20261125.md` and evidence JSON.

**Full Gate5 NOT PASSED.** Metadata only, no persistent Warp evidence payload/DS receiver fusion/watermark, no-oracle multihop routing, HELLO, native CSMA event parity, or real CUDA. No training/eval/test launched. Next priority is full network before training.

---

# 2026-11-25 -- v4.0 native DATA PDR fix and MAC trace replay

Branch: experiment/zhang-xue-v40-native-mac-trace-parity-20261125;
parent v3.9 commit 5cf397c. Ordered scope stays map -> full
network -> training/eval/test. Native ApplicationDataPacket was
previously NOT counted by NetworkMetricsFilter.on_deliver, so
v3.9 native PDR was wrong. Correct native seeds44-49:
[0,1,1,0,0,0]. Warp now differentiates app data_received
from mac_acked/delivered legacy flag. Replay genuine UavNetSim-v1
random streams: seeds44-107 show 64/64 DATA PDR match, 601 DATA
attempts and zero first-attempt timing error; 35 later starts
differ up to 550us due in part to relay app E2E ACK traffic.

Read docs/superpowers/NETWORK_V40_NATIVE_TRACE_AND_PDR_CORRECTION_20261125.md
and associated JSON evidence. **Gate5 still NOT PASSED.
Training NOT LAUNCHED**; no main branch modifications.

---

# 2026-11-25 — Zhang–Xue v3.9 network-after-map checkpoint

User-approved order: map -> full UavNetSim/Warp network/Gate5 -> only
then MATD3/MASAC training/evaluation/test. Separate branches and
commit history preserve map v3.7 SHA82edcde and MAC v3.8 SHA4a461f6.

v3.9 Warp one-hop bounded retry/partial MAC ACK and E4096 CPU
one-step map->packet->receiver-origin DS smoke created on branch
experiment/zhang-xue-v39-warp-mac-retry-20261125.
Native-vs-Warp PDR mismatches 2/6 nominal seeds; no matched native
HELLO/RNG and no persistent queues/multi-hop/E2E app ACK.
**Gate5 FAILED and training NOT LAUNCHED.** See
docs/superpowers/NETWORK_V39_BOUNDED_RETRY_BEFORE_FULL_GATE5_20261125.md.

---

# 2026-11-25 — v3.8 NETWORK AFTER MAP, STILL BLOCKING TRAINING

Order confirmed by user: MAP -> NETWORK and full Gate5 -> MATD3/MASAC
TRAINING -> EVAL/W&B/HF. Map v3.7 committed/pushed at 82edcde14fb83447ac1de3df66942af669b6136f,
separate remote branch experiment/zhang-xue-v37-warp-map-first-20261125.
Current isolated working branch experiment/zhang-xue-v38-warp-network-after-map-20261125.
Map changes are frozen on v37 snapshot and NOT modified here.

Network baseline rerun against pinned native classic v1:
6 hidden-terminal seeds, native PDR0.0 each vs collision-free Mini
reference PDR1.0 each, 56 V1 collisions. Evidence:
docs/superpowers/evidence/zhang_xue_v38_network_hidden_terminal_baseline_20261125.json.
Thus FULL Gate5 FAILS; do not run any training.

New Warp six-agent first-attempt one-hop MAC scheduler:
uav_network/communication/gpu/zhang_xue_mac_first_attempt_warp.py
combines previously independently tested DIFS/CW initial backoff,
750m CCA, data airtime, overlapping-channel SINR.
Tests/test_zhang_xue_v38_mac_first_attempt.py verify hidden same-channel
collision, channels1/6 success, near-source CCA deferral, E4096
first-attempt throughput and invalid args. E4096 x 2 DATA frames:
same-channel hidden PDR0, orthogonal PDR1, Warp CPU (no CUDA driver).
Evidence JSON in docs/superpowers/evidence/zhang_xue_v38_first_attempt_4096_cpu_20261125.json.
This is NOT a full MAC: needs frozen-backoff/retries, real MAC ACK,
queues, TTL, E2E ACK and multi-hop routing, measured parity vs SimPy.
Network gap document docs/superpowers/NETWORK_V38_AFTER_MAP_BEFORE_TRAIN_20261125.md.
Never state full network is done or allow E4096 training until gates pass.
Keep all historical branches and main untouched.

---

# 2026-11-25 — Corrected priority MAP then NETWORK then TRAINING

User clarified: first finish scientific map geometry, camera, DS and
Warp parallel consistency; next finish original UavNetSim-v1 vs Warp
network and Gate5 (including interference/retries/ACK/TTL); ONLY then
train MATD3/MASAC and run evaluation/remote logging. Do NOT start
Kaggle/W&B/HF or training before the map AND network are verified.

Active isolated branch experiment/zhang-xue-v37-warp-map-first-20261125
based on v3.6 Gate5F SHA 89c8e938db53806ef714acffccc9bd050dc6f944.
Map audit docs/superpowers/MAP_V37_BEFORE_NETWORK_20261125.md.
The prior one-cell sensor was inconsistent with a 100m-altitude,
90-degree nadir camera. It now senses every 50m cell center within
100m footprint. This adds an explicit scientific FoV assumption not
stated as a config field in v3.2 original YAML.
Shared CPU/ Warp deterministic scene sampling now generates identical
10 targets, 2 no-fly circles, 2 radio-blackout diamonds per scene.
Warp sensor/DS updates use separate private per-UAV counts and team
reward-only belief; movement kernel checks boundaries, no-fly segments
and separation. CPU/Warp parity tests and 200-seed map checks passed.
Ran map-only E4096 x 5 steps =20,480 env steps, 1,543,223 evidence
readings, 17.79565% mean coverage, all map invariants pass on
Warp CPU (no CUDA driver). Evidence JSON in
docs/superpowers/evidence/zhang_xue_v37_4096_map_fullflow_cpu_20261125.json.
No actors/replay/training or network code exercised in that E4096 run.
Next: integrate and verify Warp MAC incl. hidden-terminal collisions;
do not claim full Gate5 or GPU throughput yet. Preserve all original
source branches and do not merge into main.

---

# 2026-11-25 — Gate 5F source-executed DIFS/backoff parity

New Warp INT32 kernel at
uav_network/communication/gpu/classic_v1_backoff_warp.py
computes CW, backoff_slots * SLOT_DURATION, DIFS + backoff.
Direct source verification executed actual pinned classic V1
CsmaCa.mac_send Python generator in isolated subprocess for 128
worlds x 6 agents = 768 cases. Zero CW/window/duration mismatches.
Original V1 constants CW_MIN31, SLOT_DURATION20us, DIFS50us,
MAX_RETRANSMISSION_ATTEMPT5. Evidence:
docs/superpowers/evidence/zhang_xue_gate5f_warp_backoff_cpu_20261125.json.
Still only arithmetic with *supplied* RNG draws, NOT full
time-ordered MAC scheduler or dynamic channel-busy freeze, per-hop ACK,
retry, TTL, network queue, or Warp-GPU execution. Gate5 full end-to-end
still NOT PASSED; Gate5C hidden-terminal PDR0 vs Mini reference PDR1
remains a serious discrepancy. Next build Warp MAC event scheduler
with source-matched pending queues and random stream handling.
All earlier branch snapshots are pushed and remain rollback points.

---

# 2026-11-25 — Gate 5E actual-source CCA matching

New Warp FP32 CPU carrier sensing kernel at
uav_network/communication/gpu/classic_v1_cca_warp.py.
Directly compared to the pinned v1 original
utils.util_function.check_channel_availability (not a rewrite) across
128 random worlds x 6 nodes: 768 checks, zero mismatches; 707 busy,
61 free. Original source uses strictly < 750m and does not inspect
channel ID during CCA; hidden terminals 800m apart do not detect each
other. See evidence JSON gate5e_warp_cca_cpu_20261125.json and
tests/test_zhang_xue_v36_warp_cca.py.
Still NO dynamic CSMA scheduling, DIFS/backoff, ACK/retry, TTL or
full Warp-SimPy parity, no CUDA device. Next: source-matched
backoff scheduling and per-hop MAC retry state with corresponding
event validation. Preserve remote rollback commits and main.

---

# 2026-11-25 — Gate 5D scheduled-interference PHY on actual Warp CPU

Actual NVIDIA Warp FP32 kernel for per-receiver interference when multiple
on-air frames share/overlap channels and times, at
uav_network/communication/gpu/classic_v1_interference_warp.py.
Controlled tests: hidden overlapping co-channel SINR -0.5283353dB fail both;
non-overlapping same-channel and time-overlapping orthogonal channels 1/6
SINR 8.8819027dB pass both. Independent NumPy FP64 check of 128 random
worlds x 4 scheduled frames (512 link evaluations) within .002dB.
Test and JSON evidence: tests/test_zhang_xue_v36_warp_interference.py and
docs/superpowers/evidence/zhang_xue_gate5d_warp_interference_cpu_20261125.json.
This is NOT a full MAC or full Gate5: no CSMA backoff/CCA scheduling,
link ACK, retry/TTL, packet queue state, or E4096 GPU training.
The old mini reference still fails parity under Gate5C hidden terminals.
Keep all earlier GitHub rollback points; next implement vectorized Warp MAC
event scheduler with matched SimPy evidence. No local CUDA driver.

---

# 2026-11-25 — Gate 5C collision stress demonstrates critical parity failure

6 deterministic seeds 44–49, 2 hidden senders x0/800m -> receiver x400m,
simultaneous 62-cell DATA packets: native V1 CSMA/CA data PDR was 0.0
for every seed (56 interference/collision log events total; 10 data TX
attempts per seed), but simplified MiniDiscreteEventNet predicted PDR1.0.
This is a genuine counterexample, not an estimate of normal policy PDR.
Source evidence docs/superpowers/evidence/zhang_xue_gate5c_hidden_terminal_20261125.json
and detailed explanation docs/superpowers/GATE5_V36_PARITY_AND_BLOCKERS_20261125.md.
Real classic V1 six-UAV MATD3 and MASAC CPU smoke both ran 3 steps,
2 real learner updates; actor weights changed; short evaluations had
zero data PDR and do NOT indicate convergence.
Do not train E4096 using collision-free mini as if verified UavNetSim.
Next: collision-aware Warp MAC surrogate, then controlled SimPy-vs-Warp
PDR/hops/latency/TTL/retry comparisons; restore failure test as regression.
Previous v3.4/v3.5/v3.6 and Gate5B commits remain on independent
remote GitHub branches; current branch experiment/zhang-xue-v36-gate5b-20261125.

---

# 2026-11-25 — Gate 5B partial Warp PHY alignment

Pinned UavNetSim-v1 upstream 64bc3a24, actual native general_path_loss compared
against new Warp FP32 CPU Friis kernel on six nodes with equal Pt=0.5W,
fc=2.4GHz, N=4e-11W and SNR threshold6dB.
Measured max SNR difference 6.8073e-6 dB, mean1.3870e-6 dB, 0/36
connectivity mismatches. 400m sample8.88190269dB, 800m2.86130285dB.
Evidence docs/superpowers/evidence/zhang_xue_gate5b_warp_friis_cpu_20261125.json.
Only no-interference links on Warp CPU were tested (local CUDA unavailable).
NOT validation of MAC, interference, queues, battery, sensor, or full E4096 Warp.
Next: real collision/multi-seed vs mini and accurate discrepancy, then GitHub
incremental commit. Do not reset old branches.

---

# 2026-11-25 — Zhang–Xue v3.6 pinned classic V1 SimPy, GitHub backups

Remote branch: experiment/zhang-xue-v36-uavnetsim-simpy-20261125
Pinned upstream: EdwardS201/UavNetSim-v1 SHA 64bc3a2424f2976418731ee94f9a8fd3e8810b22, NOT newer Sionna.
Rollback: independent remote v3.4 commit 320bc5b, v3.5 commit 0803334.
Incremental v3.6 remote commits: 1ea7e1b source bootstrap, fcc3acf native SimPy real MAC/PHY, f92b863 opt-in six-agent MATD3 and HELLO-based actor peer observations.
Real native test: U0->U1->U2 DATA two hops, reverse E2E ACK, blackout recovery, no ghost Poisson or GaussMarkov jobs; MAC uses original CSMA/CA and original Greedy except staggered initial HELLO (1.5ms per ID).
Gate5A: real SimPy packet latency 1.635ms vs simplified CPU reference 1.604ms, 2hops in both, PDR 1 in both, delta31us within a 0.5ms exploratory threshold. Native Data PDR counts 1/1 even with E2E ACK. Evidence: docs/superpowers/evidence/zhang_xue_gate5a_real_simpy_vs_reference_20261125.json.
Six-agent CPU smoke: 1 train episode horizon3 and 1 eval, actor through native SimPy and real MATD3 update; ~27s runtime. No actual E4096 Warp/GPU.
Critical limits: full Warp-SimPy Gate5 still untested; native Greedy uses global destination coordinates (routing oracle) although actor peer positions are obtained from real HELLO cache; global source config not multi-env safe. Realistic collisions/TTL/retransmission across seeds to be tested and differences reported.
Next: Warp pairwise Friis parity and multi-seed network Gate5; versioned commits and remote push; no main merge and no GPU training until appropriate gate.

---

# 2026-11-25 — Zhang–Xue v3.5 CPU production-package integration

**Active branch:** experiment/zhang-xue-v35-cpu-integration-20261125
**Worktree:** /home/aduy/Documents/NCKH/uav_network_zhang_xue_v34_research_20261125
**Base commit:** 320bc5b (preserved v3.4 reference)
**Files:** uav_network/environment/zhang_xue_coop.py;
uav_network/communication/zhang_xue_ref/;
uav_network/experiments/train_zhang_xue_pilot.py
**Scope:** CPU 1000m 6UAV 10T local DS pilot with own-source packet transport,
E2E ACK, blackout, hybrid peer action, reward-only oracle, real
Vanilla MATD3 and MASAC updates using GPUHybridReplayBuffer on CPU.
**Tested:** new 16/16, original v3.4 reference 14/14, scoped
2D/M2/slotted/vanilla 159 passed/1 skipped excluding pre-existing SHA.
Random stress 60 seeds × 40 steps: 11,139 send attempts, 10,929 ACKs;
Matd3/Masac each 21 learner updates in 24 environment steps.
**Unresolved:** Full 1974-test suite is not green. Old 3D camera
FoV marginal-probability test fails on unchanged Streamfix v3 base,
as does exact old SHA fingerprint test. Existing Warp 800m/25T stays
unchanged. No true UavNetSim-v1 SimPy CSMA, GPU Warp parity, real
multi-GPU training or Kaggle/W&B/HF/GitHub push. Gate 5 deferred.
Read docs/superpowers/specs/2026-11-25-zhang-xue-v35-cpu-integration.md
before new changes. Preserve existing dirty source worktrees.

---

# 2026-11-25 — Isolated Zhang–Xue v3.4 reference fixes (NOT production)

Local worktree branch: `experiment/zhang-xue-v34-reference-fixes-20261125`, inherited from clean Streamfix v3 commit `d3e08b1`.
User deferred Gate 5 (SimPy↔Warp parity). Original dirty study/main worktrees remain untouched.
Original downloadable spec ZIP `zhang_xue_uavnetsim_v3_2_verified_20261125.zip` preserved.
Research adapter at `research/zhang_xue_v34_reference` corrects queue-full sends, packet retry after radio recovery, occupied queue never freed, safety/boundary bonus sign, FP info credit, immutable wire fields, and invalid destination admission.
Test-first red: original 7 new tests failed (5 failures, 2 errors); 2 extra packet validations failed; after code change 14/14 standalone reference tests passed. CPU PyTorch Gate4 tiny hybrid gradient smoke PASS, selected existing slotted/M2 tests 30/30 PASS, random 20 seeds × 100 packets invariant PASS. Local CUDA unavailable; training not run.
This is a **reference mini-simulator**, not installed in the production env/learner or real SimPy UavNetSim. `config_v3.2.yaml` is only the preserved reference config. Local branch not pushed to GitHub; connector write still 403.
Next: audit real actor + receiver-local belief + packet transport integration against reference; implement production tests before code; keep Gate5 deferred and no training until complete.

---

# 2026-11-21 — Paper-grounded DS threshold replaces M2 default SPRT verification

User requires reference-paper-faithful M2 detection and full algorithm flow. Primary source is user-provided Zhang et al., IEEE T-IV 2024 DOI 10.1109/TIV.2024.3352581, PDF pages 4–6 and Table III: fixed evidence mf=me=.8, per-source max positive/negative detection counts, and m(F)>=.95 target decision. Full implementation document: `docs/methods/M2_ZHANG_2024_DS_THRESHOLD_FLOW_20261121.md`.

- New M2 default `CooperativeM2Env`: `evidence_mode=zhang_max_counts`, `confirmation_mode=ds_threshold`, mf=me=.8, tau=.95, independent binary sensor rho0. No candidate/SPRT verification delay; direct DS threshold produces verified REPORT. Legacy correlated-LLR+SPRT intentionally preserved as opt-in ablation with old direct `CooperativeWorldMap` defaults for historical regression tests. Mixed profiles rejected.
- Native radio transport and GCS map delta unchanged except fix of delayed-relay new REPORT TTL creation time; `DecisionNotice.tick` remains sensing tick while the actual envelope is timestamped at generation. Report/correction revisions keep GCS consistent when DS mass crosses threshold downward/upward; not a Wald statistical guarantee.
- New tests `tests/test_m2_zhang_ds_confirmation_20261121.py` follow TDD; red tests demonstrated missing DS profile and stale REPORT creation timestamp. Verified **105 passed** targeted M2/protocol/MATD3/exact `18d756ef` parity tests (`/tmp/m2_zhang_all_targets_20261121.log`, 16.91s), no failures; the first parity attempt correctly caught an accidental legacy YAML threshold change, which was reverted before the passing rerun. Historical broader full-suite failures are not claimed fixed; full suite was not rerun. Frozen exact `18d756ef` YAML restored to original sensor.conf_threshold=.99 (which is NOT M2 DS threshold).
- Demo and standalone 1500step verification script now use rho0. Fresh 35-step Native-UavNetSim seeded demo PASS (`/tmp/m2_zhang_demo_20261121.log`, output `/tmp/m2_zhang_ds_demo_new_20261121`): GCS coverage .11125, 612 delivered evidence events, 244 radio packets received, 1 correct target reported. **Diagnostic scripted policy, not RL result; full 1500-step scenario NOT rerun after this code change**. M2 Native MATD3 GPU learner pilot code uses rho0 and semantic checkpoint format/schema v2; all old v1 SPRT policy files remain distinct. NO GPU Warp full-M2 training, Kaggle launch, or user history overwrite. Current source and tests refer to the **isolated clean M2 worktree**, never reset or clean the dirty legacy source.

---

# 2026-11-20 — Episode clock, reset and E4096 follow-up audit

User asked whether the 4096-episode cliff is plotting, parallel collection, reset or learning. Reused all 402 hash-verified HF files and read original Spinning Up, BenchMARL, CleanRL, SB3, MAPPO, Ackermann MATD3 and FastTD3 sources.

- All three MATD3s: episodes 1..4096 finish simultaneously at 6,144,000 transitions; only FIVE distinct completion clocks through 20k retained episodes. Episode 4097 completes at 12,288,000, not immediately after reset. Worker order within one batch is not policy progress.
- Actual W&B history has 626 sampled episode-return points/run (episode 1 + multiples of 32). All 2,410 points intersecting readable HF rows match both return and completion global_step EXACTLY. 91ba54dc still lacks 2,980 HF episode rows; its 94 extra sampled W&B rows cannot reconstruct them.
- Every MATD3 has 7,500 reward + progress + optimizer journal rows, no duplicate/gapped global steps. For 12 complete 4096-episode groups, integrating 1500 per-vector mean rewards reproduces episode-return means with error ZERO.
- Source-based reset probes on all four historical configs (three exact frozen bundles), CPU 4-env / horizon3 / H32 / B4: 44/44 invariants pass after real Adam updates. Model/target weights, Adam moments, learner counters, replay rows/index/size/RNG and Torch RNG persist. Terminal next observations survive reset and next step; 3-D discovery/delivery offsets match nonzero world-info sentinels. This is not a GPU convergence test.
- Orthogonal reset log: step1500 reward -3.8964, next step +2.0370; battery returns ~100%, live progress restarts. Actor/critic update counts increase across reset.
- MATD3 learning failure remains after replot: independent 300-world validation return orthogonal -2909.59 at checkpoint0 -> -3249.15; default U4 -2627.53 -> -3250.07. Orthogonal first sampled simultaneous actor-gradient/tanh-derivative zeros already at 3,014,656 transitions BEFORE first reset. Report distinguishes measured saturation from unproven initiating hyperparameter.
- MASAC same E4096 recovers, final actor gradient 0.0387445. Legacy reward validation namespace is gpu_eval/mean_return + gpu_eval/checkpoint_global_step (30 points), not the absent gpu_eval_periodic/mean_return alias. Best periodic validation return -735.11/delivery35.74% at 22,937,600. Do not fill missing namespace values with zero.
- E4096 changes scaling: warmup131072 = 32 world steps; replay1M ~244 vector steps; U8/B8192 = 16 sample draws per new joint transition in steady state. Neither learning before full episode nor replay shorter than horizon proves a bug. Reducing E4096->256 while keeping U8 multiplies update density 16x; an E-only causal claim requires controlled update/frame budgets.
- Framework nuance: SB3 supports episode X; MAPPO uses nested episode/step but logs total_num_steps; BenchMARL backend logger uses iteration, counters/total_frames and evaluation JSON step_count use frames; FastTD3 global_step is vector iterations and frame is global_step*num_envs. Do not declare episode budgets inherently incorrect.
- New standalone offline deliverable ~/Downloads/UAV_PLOT_RESET_AUDIT_20261120/: Vietnamese report, four PNG/PDF figures, exact scalar exports/CSVs, replot.py, verify_export.py, portable probe_reset.py + configs. Replot/verify pass; independent scalar accounting max error0, float64 CSV aggregation roundoff <5e-13. No online mutation or training job.
- Existing algorithm changes from previous task remain uncommitted; this follow-up changes NO learner loss/config/dashboard. Do not call the prior optional .001 output regularizer a proven full-training cure.
- No E/replay/warmup GPU ablation, multi-seed training, or new full Kaggle run executed. Next: use fixed-policy validation vs checkpoint transitions as main learning curve; treat completed episode order and live mission state separately. Keep user 20k-episode stop rule; compare algorithms at common observed transition budgets.

The commit for this note is documentation-only; preexisting dirty work is preserved.

---

# 2026-11-20 — Four-run saturation audit and matched source-reference baselines

Verified at 2026-11-20T12:14:55.887896+07:00; base source commit e346e26a4e595b725736ab521f994862acda7a65.
User requested live W&B/HF analysis, source/paper audit, fixes and a fair MATD3/MASAC pair.

- W&B/HF checked for 653793e0, 91ba54dc, 4d398c22, 42980408. 402 HF files verified against LFS SHA256 or Git blob SHA1; raw episode counts 20000/17020/20000/20000. Missing 2980 HF episodes in 91ba54dc; never impute them.
- Three MATD3 Actors lose gradients with saturated tanh. Orthogonal already has sampled zero Actor/motion derivative at 3,014,656 transitions, before its first completed cohort at 6,144,000. All four runs already have Actor LayerNorm enabled.
- MASAC training delivery cohort means 37.73 -> 29.30 -> 39.33 -> 43.44 -> 46.40%; final Actor gradient 0.0387445. Deterministic validation delivery 0.66% at ~6M to 35.74% at 22,937,600. Do not call this the same Actor freeze.
- Original Ackermann TF1 MATD3 p_train includes 1e-3 mean(flatparam^2), absent in UAV. Added opt-in actor_output_regularization (legacy default zero); new matched profile/notebooks enable .001. This is an author-source referenced hybrid action adaptation, NOT a universally required TD3 formula or proven cure for full training collapse. Q1-only TD3 Actor remains correct; no NormXYZ, reward/physics/observation/safety change.
- New common guard compares all 13 OffPolicyConfig fields and default initialization. Pair: E4096/U8/B8192/H256x512, shared Actor/Critic, Actor LN true / Critic LN false, replay1M, seed44, 20000 completed episodes, validation300/test4000. MASAC dense reference path; entropy/temperature retained.
- Both NEW notebooks in ~/Downloads/UAV_MATCHED_AUTHOR_REFERENCE_S44_20261120 embed source SHA256 24535a8655a216c3bfa920057ce6964dd8eb5e59582d73152aea68be7f50566a. Their parent is afe4f9c98c5abd1a6f9c0efc46261803bf800cfe52c58588713ee0c497285eca. Original notebooks unchanged.
- 25 new tests pass in repo and extracted notebook source. Final related suite 118 passed, 2 skipped, 3 subtests passed. Both actual CPU notebook preflights pass. Four coefficient-zero CPU updates are bitwise equal to before patch.
- Full tests/ run initially 1725 passed,40 failed,105 skipped,1984 subtests passed. Updated two historical fixtures to omit the newly introduced config field; related rerun passes. Remaining 27 failed test/subtest cases reproduced exactly on BEFORE-PATCH source too. See outputs/baseline_rootcause_20261120/test_status.json and logs. Never claim entire repo green.
- Current repo lacks deferred_gpu_eval_keep_all, so its local matching profile uses target/candidate 1000/2000; notebooks retain the newer keep-all DenseEval implementation. Do not silently replace frozen notebook source with local HEAD.
- No Kaggle Training/Validation/Test launched, no W&B/HF mutation, no credentials exposed. Full GPU performance, monotonic reward, improved delivery and one-session completion are UNPROVEN.
- Code and test edits remain in the preexisting dirty working tree (325 paths were already modified). This commit records continuity only, not a clean published learner build. Backups and minimal session-start diff: outputs/baseline_rootcause_20261120/.
- Read the delivered report UAV_MATD3_MASAC_ROOT_CAUSE_AND_MATCHED_BASELINES_20261120.md and verification_manifest.json before next baseline task. Repro builder: scripts/research/build_matched_author_reference_baselines_20261120.py. Preserve user's outstanding edits; do not reset/clean.

---

# 2026-11-16 — Restored W&B Personal Workspace after tracker startup

MASAC valid_routes run df258798 started at 15:37 ICT and was still training
when the Runs table was reported changed. The notebook's embedded
OnlineTracker called ensure_project_workspace at startup. Its runset
reconciliation reproduces the exact replacement of the user's 13 columns
with 17 default columns. A separate post-training command targeted a saved
view. The two paths were removed from the delivered notebook.

Restored the same Personal Workspace nw=nwusertruongdaoanhduy on the W&B
server using the 01:37 ICT snapshot and the current personal run selection.
Readback verified 13 Runs columns, 14 Training plots, 9 Eval plots, 9 Final
Test plots and zero run filters. Before/after specs are stored under ignored
outputs/research/wandb_personal_restore_20261116.

Notebook was updated at the original path in Downloads. Embedded SHA256
1c1e59e200a6ed1e0481a86fae24a788c708f652e919be18aae93d3fe57e90a7.
It still logs to the run's W&B history and still runs GPU Validation/Test.
The in-progress Kaggle run used the previously imported notebook archive.
Tracker fix 3141e96d1bd4be5b75bc94ba0ff7d43a58c6f995. Offline regression reproduced the startup write before
the fix, and the tracking/dashboard suite passed 33 tests after it.

---

# 2026-11-16 — Four W&B baselines and MASAC valid-route Twin Q

Read W&B live in the Personal Workspace and through the Public API. All four
seed-44 runs finished 20,000 training episodes, with 100 held-out Final Test
scenarios (base seed 2000000044).

| Run | W&B elapsed | Transitions | Learner updates | Discovery % | GCS delivery % |
|---|---:|---:|---:|---:|---:|
| MASAC dense `f57a5e0f` | 09:41:00 | 24,485,888 | 47,576 | 56.26 | 54.31 |
| MASAC ACCEL `d12f350a` | 06:01:56 | 25,079,808 | 48,736 | 47.29 | 45.51 |
| MATD3 dense `49a76aae` | 02:22:32 | 24,141,824 | 46,904 | 56.09 | 52.97 |
| MATD3 ACCEL `3facc656` | 02:09:27 | 22,466,560 | 43,632 | 57.17 | 52.86 |

At exactly 17,891,328 validation transitions MASAC GCS delivery was 44.46%
dense and 38.21% ACCEL. Training-history mean discovery/delivery also fell from
50.34/47.52% to 45.28/42.60%. Both used the same recorded scientific settings.
The original and ACCEL notebooks recorded the same git reference but contained
different source archives. The MASAC shared critic replaced a dense affine
projection with two projections and a broadcast addition, in both the actor
and soft Bellman target. FP32 numerical differences are reproducible. This
identifies a plausible learning-dynamics mechanism. Its effect on final
performance still needs a controlled T4 comparison. The captured validation
candidate counts were 129 and 96, respectively. Final test scores keep their
original W&B provenance.

Implementation commit `d1ee37159495220a8f4a1e5fab754c04cf5f8320` adds
`masac.shared_critic_mode=valid_routes`. Both learner paths retain the dense
Twin Q on identical state/joint-action features, evaluate only the observation
mask's supported discrete routes and retain the original weighted reduction.
`dense` is the source default with legacy fingerprint compatibility.
`split` is the explicit archived ACCEL numerical path. The loss, hyperparameters,
mask-aware temperature and replay sampling are carried over.

CPU test on 500 recorded random-policy Warp transitions from the 35T25O
scenario with six UAVs, B1024 and two threads. 14.75% of route candidates were
valid. Median full learner update was 4.17 s dense, 2.33 s split and 0.84 s
valid_routes across six timed iterations per mode. On the common minibatch
and weights, max absolute Q delta to dense was 2.98e-8 for split and 7.45e-9
for valid_routes. Seven matched updates yielded zero measured deterministic
action RMS delta for valid_routes on this replay. These measurements are CPU
diagnostics. Worktree regression: 90 passed, 2 CUDA skipped, 43 subtests.
Isolated git candidate: 71 passed, 2 CUDA skipped, 43 subtests.
Extracted notebook regression: 4 passed, 2 CUDA skipped. Notebook schema,
resolved config and embedded archive digest verified.

Kaggle UI notebook in Downloads
`MASAC_E4096_U8_B8192_H256x512_VALID_ROUTES_TRAIN11H_20261116.ipynb`.
E4096/U8/B8192, R1M, shared H256x512 with Actor LN, seed 44, 20k episodes,
39,600 s training cap and 42,600 s session cap, two-GPU Validation and 100
held-out Final Test. Embedded source SHA256
`1907be062e9ae6c58cc463f819df2bc38e1810cae6233c0f652fe35b1fb90146`.
The notebook records execution mode and source bundle/patch hash in W&B.
Existing notebooks and W&B runs were preserved. No Kaggle training was
submitted in this audit.

Next validation is a manual Kaggle T4x2 run of valid_routes, with comparisons
to dense and ACCEL at the same number of transitions, fixed validation and
held-out scenario seeds, recorded learner wall time and peak VRAM. Keep the
historical dense MASAC/MATD3 pair as the measured reference until this
comparison. Assess paper-level uncertainty across independent training seeds.

---

# 2026-11-08 — Q1/A* MASAC/MATD3 source-code paper audit

Full source-backed audit:
`docs/methods/Q1_ASTAR_MASAC_MATD3_PAPERS_AND_SOURCE_CODE_AUDIT_2026-11-08.md`.
Open-sourced reference clones at gitignored
`outputs/reference_source_audit_20261108/` (NOT vendored to UAV).
BenchMARL official JMLR2024 Q1 MASAC
`facebookresearch/BenchMARL` pin `65d649d` successfully
trained on CPU VMAS Balance 4 agents, 192 frames/6 iterations,
seed44, with alpha/actor/critic CSV. Reproducer:
`scripts/research/run_benchmarl_masac_cpu_smoke.py`; evidence:
`docs/methods/evidence/reference_benchmarl_masac_smoke_2026-11-08.json`.
**Not UAV 35T25O training** and no direct hybrid XYZ/RF+discrete mask
support: BenchMARL branches `SACLoss` (continuous) vs
`DiscreteSACLoss` (discrete) rather than their simultaneous hybrid.

Upstream `marlbenchmark/off-policy` MATD3 pin `41fd5eb` has
`self.q_outs=[...]` Python list: both Q heads absent from
`critic.parameters()`; actual 3-step Adam probe confirms both
Q output weights frozen despite gradients. Temporary **in-memory**
`nn.ModuleList` fix makes heads update. Reproducer/evidence:
`scripts/research/verify_upstream_matd3_critic_registration.py`,
`docs/methods/evidence/reference_matd3_q_registration_2026-11-08.json`.
**NOT a bug in user's UAV TwinCritic**: direct local test verifies
both registered and both Q heads updated.

Original MATD3 author Ackermann official TF1 2019 NeurIPS **workshop**
`JohannesAck/MATD3implementation` pin `fd6c7d0`; author TF2
implementation `JohannesAck/tf2multiagentrl` pin `d5442f6`
contains both MASAC/MATD3 but MASAC lacks automatic temperature and
Gaussian policy implementation (TODO in source), README acknowledges
MATD3 performed better in their MPE tests. Hybrid MASAC CoRL2025
Fu paper useful mathematically but no official runnable repo verified.
AAMAS2026 AC-MASAC Liu UAV has real training code pin `771465c`,
but **ICORE2026 rank A** (CORE2023 A*, downgraded); do not cite
as current A*. ECCV2024 Qiao CCE paper A* linked repo has
README/LICENSE only, no training code. Single-agent ICML2018
A* SAC Haarnoja and TD3 Fujimoto are original citations;
CTDE from NeurIPS2017 MADDPG Lowe; MATD3 primary Ackermann.
No Kaggle GPU/full 20k UAV training launched or learning code changed.

---

# 2026-11-08 — MASAC temperature loss vs mask-aware root-cause audit

Completed read-only W&B review and controlled same-batch Warp 35T/25O
CPU micro stress test. Report:
`docs/methods/MASAC_TEMPERATURE_MASKAWARE_CAUSAL_AUDIT_2026-11-08.md`.

**Strong paired W&B 20T/15O evidence:** same original source commit
`4c9f67c`, same seed 44 and science settings; `62cfc303`
(mask-aware=false) best delivery **1%**, discrete alpha
**60,630,352**, continuous alpha **27,961**; `f6a196a7`
(mask-aware=true) best delivery **61%**, alpha_d **0.977**,
alpha_c **0.0103**. Mask-aware is CRITICAL for masked UAV routing.

**Not the reason 61% vs 52% differs:** Both `f6a196a7` (20T/15O,
loss=log_alpha) and `7f5fd73f` (35T/25O, loss=alpha)
have mask-aware=true. Original Actor observation dimension changes
**198 to 228** (shared Actor 204 to 234); critic state
**565 to 830**. Best validation metrics: old 20T search67%,
delivery61%, CBF0.577; new 35T search54.286%, delivery52%,
CBF0.674. New run actually delivers 18.2 targets/episode
vs 12.2 old; fraction conditional on found improves
91% to 95.8%.

No full-budget exact same-scenario alpha-vs-log_alpha W&B pair
exists for MASAC 35T/25O. Do NOT attribute the 9 percentage
point rate drop to the temperature formula.

**Executed real-code paired CPU replay stress:** 35T25O Warp
260 rows, fixed starting Shared-MASAC LayerNorm weights,
same minibatches, one config variable `temperature_loss`,
H64 B64, 1200 `MASAC.update` steps.
Results `log_alpha` alpha_c **0.69728** / alpha_d **0.88730**,
vs `alpha` alpha_c **0.71512** / alpha_d **0.89106**,
heldout deterministic action RMS difference **0.00429**.
This is small-scale off-policy, NOT 20k end-to-end validation.
Evidence and script:
`docs/methods/evidence/masac_temp_35t25o_paired_cpu_1200upd_2026-11-08.json`,
`scripts/diagnose_masac_temperature_35t25o.py`.
No Kaggle job/training launched, no learning equations edited.
Next causal full run if user requests: same 35T25O source/seed/architecture
with only temp=log_alpha vs alpha at equal global-step checkpoints.

---

# 2026-11-08 — MASAC reviewer-grade correctness + sampled-vs-mean probe

**New audit:** `docs/methods/MASAC_REVIEWER_BASELINE_CORRECTNESS_AUDIT_2026-11-08.md`.
Code checked against original SAC, Hybrid SAC, Fu Hybrid-MASAC and
MATD3 source; checked math updates continue to pass tests; **no proven
formula bug or loss rewrite**. Do not claim implementation is
author-paper-exact (no Fu attention critic). No training started.

**Actual existing best MASAC Actors** verified against saved HF SHA and
run on identical 5 validation seeds in four policy modes (only
inference): `scripts/diagnose_masac_policy_modes.py`,
`docs/methods/evidence/masac_four_policy_modes_2026-11-08.json`.
Early H512/U16 `c70aa6f4`: mean+argmax **18.86%**;
Gaussian-sampled+argmax **33.14%**; mean+sample-destination
**21.71%**; sample-both **37.14%**. Thus primary
early sampled-vs-deterministic gap is continuous movement action,
not categorical routing. Late H256/U8 `7f5fd73f`:
deterministic **57.71%** on local replay (original W&B best
**52.00%**), other three modes 51.43%, 51.43%, 53.71%.

**W&B source matched-global-step 7.37M:** CBF interventions per
UAV-step MASAC H256 **0.932**, MATD3 H256 **0.610**;
MASAC H512 **0.977**, MATD3 H512 **0.804**.
Search and delivery both very low in early MASAC, improving
as intervention declines; correlation not proof of causation.
MASAC H256 best W&B **52%** vs MATD3 H256 best **51.43%**
at unequal checkpoint budgets; do NOT write that MASAC
always scores lower.

**Critical unresolved reproducibility finding:** W&B run `7f5fd73f`
original per-seed metrics.json was read via W&B (not rebuilt).
**5/5 evaluation seed initial layout hashes match** current CPU
replay with same SHA-verified saved actor, but original W&B
**91/175 delivered (52%)** vs current local **101/175 (57.71%)**;
some episode lengths/trajectories diverge. Evidence:
`docs/methods/evidence/masac_wandb_local_same_seed_replay_2026-11-08.json`.
Do not silently replace W&B metrics with local replay values.
Source/environment code diff from recorded training commit
`a6e07a8f4b199028298b6ffd9a47f2cecd654fc9` contains
no environment/communication changes; numerical runtime,
inference and simulator determinism need further isolated diagnosis.
Next requested task should be **one** no-training
cross-runtime first-divergence trace (first 10 actions and
positions in pinned Kaggle-like CPU environment), not any
relaunch of the old 10 W&B runs.

---

# 2026-11-08 — PAPER-CORE CTDE HYBRID BASELINE CONTRACT + 2 UI BUILDERS

**User asked for an actual source-faithful baseline, not another
posthoc W&B recommendation.** Created separate strict 35T25O config
`configs/comparison-paper-core-35t25o-xyz-sensing.yaml`:
6 independent local Actors, 6 independent centralized twin Critics,
No Actor LayerNorm, no PQL, mask-aware/alpha MASAC, MATD3 original
delay2/twin min/Gaussian smoothing/replay peer actions/default init;
H512×512, U8, B8192, replay1M, seed44. World/CBF/reward/network
identical to preexisting independent H512 scenario. Not source-paper
byte-exact because the domain has hybrid XYZ/RF + masked destination
and uses full global CTDE state for Critics.

**Prepared verification** `scripts/verify_paper_core_baseline.py`,
offline contract JSON `docs/methods/evidence/paper_core_35t25o_contract_2026-11-08.json`,
unit/math/negative tests `tests/test_paper_core_35t25o_baseline.py`,
and UI spec tests `tests/test_paper_core_ui_notebooks.py`.
Added opt-in `paper_core=True` to existing
`uav_network/cloud/ui_notebooks.py`; default legacy notebook
generation remains unchanged. `scripts/build_paper_core_35t25o_ui_notebooks.py`
writes two new source-pinned UI notebooks **after commit**. Do not
submit any training automatically. Historical ten W&B experiments
remain archived, unmodified, and are NOT the same strict-U8-NoLN run.

**Actual algorithm equations:** audit
`docs/methods/PAPER_CORE_VS_EMPIRICAL_BASELINE_35T25O_2026-11-08.md`
with source references for MASAC and MATD3. Since original code's
math update functions passed source-oriented numerical tests
and source commit comparison revealed only comment/docstring edits,
we have NOT modified the algorithm loss/smoothing/discrete logic.
Do not claim an exact copy of Ackermann's original MPE code or
Fu's attention critic, or that shared/LN belongs to the
paper-core architecture. Existing Shared H512 U8 ActorLN MATD3
`ccaadcd6` remains the empirically best tested 20k run and
MASAC Shared H256 U8 ActorLN `7f5fd73f` a completed measured
baseline; neither run relabeled.

**Next step if requested after this:** full-strict source-aligned
baseline experiments are new, distinct 2-run cohort, potentially
computationally expensive, only when explicitly approved. Do not
run the same earlier ten jobs again. Retain W&B/HF evidence and
do not rewrite loss based only on Q(soft)-Q(hard) speculation.

---

# 2026-11-08 — COMPLETED: checkpoint-based MATD3 Gumbel/hard action support diagnostic

**Task closed without algorithm change or new training.** After the completed
10-run W&B analysis, inspected live original W&B model artifacts and pinned
Hugging Face best snapshots `ccaadcd6` (MATD3 Shared H512 U8 ActorLN)
and `635d7134` (Shared H256 U8 ActorLN). Both checkpoint files
have verified SHA256 and type `purpose=evaluation_only` with only
`actors` tensors (12), zero trained Critic tensors. The W&B model artifacts
have only `checkpoint.json` pointing back to these same HF best policies.
**A real trained Q(soft)-Q(hard) result is therefore not available from
the saved best-checkpoint evidence.** No random Critic Q numbers were used.

Executed read-only rollout of the exact Warp CPU backend on the 5 fixed
validation seeds 1000000044..1000000048, deterministic trained Actors,
one Gumbel(tau1,hard=False) per observation and one-hot of same argmax:
- `ccaadcd6`: 32,484 UAV observations; 338 with >1 valid destination
  (1.04%), soft-hard eligible L1 mean 0.03724, max 0.86058.
- `635d7134`: 32,544 observations; 348 with >1 valid destination
  (1.07%), soft-hard eligible L1 mean 0.01076, max 0.78315.
Both show mostly forced-silent masks. Action mismatch is **not** Q error.

Read the detailed
`docs/methods/MATD3_TRAINED_ACTOR_GUMBEL_SUPPORT_AUDIT_2026-11-08.md`
and numeric evidence
`docs/methods/evidence/matd3_trained_actor_gumbel_support_2026-11-08.json`.
Reproduction script `scripts/diagnose_matd3_gumbel_support.py` and test
`tests/test_matd3_gumbel_support_diagnostic.py`. Original downloaded
Actor checkpoints and raw outputs stay local under
`outputs/wandb-ten-20261108/diagnostics/`; do not commit weights or secrets.

**Current decisions:** Shared Actor + Shared Critic, Actor LayerNorm on
but no isolated Shared LN benefit evidence, Critic LayerNorm off, U8 over
U16 for the studied 35T25O/seed44 setup. Do NOT change MATD3 Gumbel or
other algorithm mathematics. Do NOT repeat the original ten training runs
unless scientific code/config/scenario changes; no run started here.

**Next research task if user explicitly requests:** look for a *real
historical full-learner critic checkpoint* if any (do not confuse
W&B best actors with trained critics); otherwise study the existing
delivery and admissible-link/mask diagnostic metrics without training.
Any future optional critic-state snapshot must be a newly named
instrumentation capability, not a silent change to baseline objectives.

---

# 2026-11-08 — COMPLETED live comparison of 10 latest 35T25O BASE W&B runs

**Current task finished; do not retrain duplicates.** Read
`docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md`,
the machine-readable `docs/methods/evidence/wandb_latest_ten_2026-11-08.json`,
and `docs/methods/evidence/wandb_10_case_reuse_registry_2026-11-08.json`
before discussing MATD3/MASAC 35T25O runs.

Read-only direct W&B API snapshot on 2026-11-08. Ten most recent BASE runs:
MATD3: 01 `ccaadcd6` (20k), 02 `f55774e2` (20k),
03 `55158605` (16384), 04 `baf27732` (20k),
05 `635d7134` (20k). MASAC: 01 `d9c1c5e9` (11675),
02 `c70aa6f4` (4096), 03 `70eaad3a` (4096),
04 `c6cf7916` (4096), 05 `7f5fd73f` (20k).

All ten W&B source commits = `a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`;
world/sensing/network/energy/reward scientific SHA-256 and runtime
protocol hashes match across all ten. Only five reached target 20k,
despite W&B state `finished`. At all-run common comparison point:
last 1024 training episodes through episode4096; independently,
5-episode fixed-seed validation at checkpoint transition 7,372,800.
Pairwise comparisons instead use each pair's minimum completed
episode and latest exact shared validation checkpoint. No future-point
lookahead; best checkpoints across unequal training horizons are
not a fair same-budget ranking.

**Observed decision** within seed44, 5-episode validation:
MATD3 Shared H512 U8 ActorLN `ccaadcd6` is primary
(best validation delivery58.86%, final58.29%, W&B lifetime2.63h);
MATD3 Shared H256 U8 ActorLN `635d7134` is faster
(20k in1.74h, best51.43%); MASAC Shared H256 U8 ActorLN
`7f5fd73f` is the only completed 20k MASAC
(best52.0%, lifetime8.33h). MATD3 Shared H512 U8 beats
Shared H512 U16 by13.71 delivery percentage points at same
23,961,600-transition checkpoint, and the U8 run completed faster.
Independent U16 MATD3 substantially underperformed Shared U16
under pairwise matched comparisons. MASAC H512 U8 improved later but
stopped at 11,675, so do not claim a 20k ranking for it.

**Best checkpoint metadata:** all ten W&B-reported Hugging Face pinned
revisions and `uav-network/<run_id>/best/checkpoint.pt` file entries
verified read-only on 2026-11-08, see
`docs/methods/evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json`.
Full checkpoint bytes were not downloaded/hashed.

**IMPLEMENTED:** Read-only capture/analysis script
`scripts/audit_latest_ten_wandb_baselines.py`; offline
report/figures/registry script
`scripts/render_wandb_ten_case_registry.py`;
regression test `tests/test_wandb_ten_run_audit.py`.
No Kaggle training, W&B changes, HF mutation or scientific code changes.
Raw history cached locally under `outputs/wandb-ten-20261108/`
and dated copy in `~/Downloads`. Next chat must reuse these records
instead of re-running identical experiments. If sources or scientific
settings materially change, treat as a new cohort with its own fingerprint.

---

# 2026-11-08 — Spinning Up SAC/TD3 → CTDE Hybrid-MASAC/MATD3 source audit

User authorized proceeding from a research-only review to an implementation/audit phase,
with a one-task-at-a-time rule and no unverified baseline algorithm or reward changes.
Work verified in sandbox at initial HEAD `cc1885f` on
`experiment/hfbest-embedded-source-steps1500-20261105`. Initial worktree was clean.

**IMPLEMENTED in commit `ec2e069`:**

- Added `tests/test_paper_faithfulness_audit_20261108.py` (5 executable regression
  tests) for MASAC shared-critic exact conditional expectation against a nonlinear
  joint-Q reference, MATD3 target noise on all UAVs in both parameter-sharing
  topologies, soft-vs-hard categorical critic input, and replay command under CBF.
- Added source-level provenance comments to `masac.py` and `matd3.py`, without
  modifying their mathematical expressions or training behavior.
- Added `docs/methods/SPINNINGUP_SAC_TD3_MARL_AUDIT_2026-11-08.md`
  with paper/source findings and the remaining research caveats.

**VERIFIED:**

- Reference: OpenAI Spinning Up SAC/TD3; Ackermann MATD3 2019 Eq. (5).
  MATD3 paper smooths **all** agents' target actions, as current implementation
  does, even though one author TF2 implementation smooths only the learner's own.
- Shared-MASAC categorical estimator matches the exact joint categorical
  expectation on a non-additive synthetic Q under a factorized joint policy.
- Soft Gumbel Q inputs and hard replay one-hot are a documented representational
  difference, **not** a proven correctness bug or reason to change baseline.
- The MDP action stored in replay is the issued pre-CBF command; safety filtering
  lives in the environment transition. A CPU boundary test confirms this.
- Regression command:
  `python -m pytest -q tests/test_paper_faithfulness_audit_20261108.py tests/test_algorithm_math_regressions.py tests/test_reference_baseline_contract.py tests/test_reference_critic_gradients.py tests/test_parameter_sharing.py tests/test_matd3.py tests/test_masac.py tests/test_split_updates.py tests/test_xyz_cbf.py tests/test_xyz_cbf_warp.py tests/test_masac_ui_ablation.py tests/test_matd3_ui_ablation.py`
  → **118 passed, 43 subtests passed**. `python -m py_compile` and
  `git diff --check` also passed.

**NOT CHANGED:** No policy/critic loss modification, no Gumbel estimator switch,
no reward, world, CBF, observation, replay schema, seed 44, hyperparameter,
notebook, Kaggle execution, cloud storage or checkpoint alteration.
No full GPU-training benchmark was run.

**NEXT ACTION IF REQUESTED:** For a fixed trained checkpoint/replay minibatch,
measure MATD3 Q(soft Gumbel) versus Q(hard one-hot) and assess any downstream
bias. Consider any straight-through estimator only as a separately named
ablation; do not silently modify BASE.

---

# 2026-11-08 — Double validation cadence BEFORE starting 10 Kaggle UI runs

User authorized changing validation frequency immediately instead of waiting for training results. Started ~08/11/2026 02:42 UTC+7; verified 08/11/2026 02:46:28 UTC+7. **Active 20k baseline configurations now use `tracking.evaluation_interval_steps=1843200`**, exactly half of old 3686400. Changed only three current profiles: `comparison-independent-35t25o-h512-xyz-sensing.yaml`, `comparison-independent-baseline-xyz-sensing.yaml`, `comparison-reference-masac-matd3-xyz-sensing.yaml`. Did not change old/historical 20k/25k profiles. Kept 5 periodic validation episodes, fixed seeds 1000000044..48, evaluator timeout 900 s, best criterion, CPU async evaluator, checkpoint cadence256 and max20k episodes. No reward/algorithm/CBF/world/observation/scientific changes.

Main source code commit **`a6e07a8`** embeds the new cadence into both MATD3 and MASAC Kaggle UI intros + active YAMLs; test expectations updated. **105 passed**, including baseline MATD3/MASAC notebook tests, async evaluator and controlled study; historical `test_controlled_study` remained on original historical profile. Git diff --check clean.

**Re-exported all 10 Kaggle UI `.ipynb` files** on user's laptop `~/Downloads` and in repo ignored `outputs/kaggle` using the updated source commit `a6e07a8`. Replaced same-name earlier `.ipynb` files plus old-name ZIP to avoid accidental stale imports. Created clearly labeled ZIP: `~/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_35T25O_S44_VAL1843200.zip` (same copy in `outputs/kaggle`), with manifest `~/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_MANIFEST.json`. Verified actual embedded archive YAML values=1843200/5/256, independent case recipe overrides, PQL disabled, compiled notebook cells, schema, 10 ZIP entries, source integrity and copied notebook byte-for-byte. Notebook source commit `a6e07a8` is intentional snapshot even if subsequent continuity note commit advances HEAD. The downloaded files are **self-contained and sensitive source**, never publish to public git. User must import/reimport updated `.ipynb` in Kaggle; previously imported Kaggle notebooks will **not** auto-update. No training submitted and no W&B/HF mutation.

For old actual MATD3 BASE ~21.19M transitions the new interval gives 11 theoretical validation threshold crossings vs 5 old; at max30M transitions new16 vs8 old. These are thresholds, not guaranteed successful validations: single CPU evaluator may skip_busy, deadline, fail/timeout, and terminal threshold may be skipped. Future work only if user requests changes to busy-skip or evaluation method; avoid modifying algorithm/reward or validation protocol otherwise.

---

# 2026-11-08 — Export 10 Kaggle UI notebooks for user (MATD3 BASE x5, MASAC BASE x5)

User explicitly requested stopping reward plotting/changes and preparing (without running) **exactly 10 manually importable Kaggle UI `.ipynb` files**: MATD3 5 architecture ablations and MASAC 5 identical architecture/update cases. Start ~08/11/2026 02:25:48 UTC+7, verified 08/11/2026 02:33:00 UTC+7.

**Implementation commit `89e6274`**, pinned inside every notebook. Only added MASAC Kaggle notebook builder, 10-case export script, tests and guide. No changes to `reward`, algorithm update equations, CBF, world, sensor, episode aggregation, training code, MATD3 builder, validation cadence or checkpoint logic.

Matrix for both MASAC & MATD3: case01 SH H512 U8 ActorLN; case02 SH H512 U16 ActorLN; case03 IND H512 U16 ActorLN; case04 IND H512 U16 ActorNoLN; case05 SH H256 U8 ActorLN. Both Actor and Critic share/independent **across UAVs**, not with each other. Critic never has LN (existing code option actor only). MASAC uses mask-aware target entropy and alpha loss, MATD3 default initialization + policy_delay2. BASE only; PQL disabled.

Actual 10 exported files and one 6.9MB ZIP are ON THE USER'S LAPTOP under `/home/aduy/Downloads/`, duplicated in `/home/aduy/Documents/NCKH/uav_network_study_1200/outputs/kaggle/`; zip exact name `UAV_MATD3_MASAC_10_KAGGLE_UI_35T25O_S44.zip`. Local SHA/file verification manifest in `/home/aduy/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_MANIFEST.json`. **Never commit/embed these private notebooks into public repository** — they include full bundled source for manual Kaggle UI, and anyone with notebook access can inspect them.

Source bundle SHA256 `f28cd52ef906e5623a592bffab0822a4b8406bceef5e78de11d89c8ceef96b28`; extracted and compared **all 429 bundled source files byte-for-byte** against repo; validated 10 nbformat schemas, compiled all code cells, full runtime `SPEC` from actual launch cell and reward equality; duplicated Downloads vs outputs/kaggle content hash verified; ZIP contains exactly 10 files and testzip passed. Current full config: 6 UAV, 35 targets, 25 obstacles, 1km, 1500 max steps, 20k episodes, seed44, 4096 env, batch8192, warmup131072, replay1M 75%CPU/25%GPU, no chunk, no compile/AMP/fused, deterministic true, checkpoint every256 episodes, fixed validation 5 episodes per3,686,400 transitions, best HF + W&B online. Manual Kaggle UI private, Internet on, T4x2, Secrets WANDB_API_KEY/HF_TOKEN and edit NOTEBOOK_REF. Kaggle remote runner max nominal 39600s minus cleanup and startup, possibly stops early.

Tests: **70 passed** (`tests/test_masac_ui_ablation.py`, `tests/test_matd3_ui_ablation.py`, reference/scaled UI checks, checkpoint retention). MASAC genuine U16 independent learner updates on CPU passed. No GPU training launched and no Kaggle/W&B/HF resources mutated. Next user action is private Kaggle notebook import and Save & Run All. Notebook source snapshot is pinned to `89e6274`; subsequent note-only commits do not change its program source. Source guide: `docs/methods/MATD3_MASAC_10_KAGGLE_UI_ABLATIONS.md`.

---

# 2026-11-08 — User-selected MATD3/MASAC BASE Reward plots at windows 8/16/32/64

User corrected earlier 'latest' run preference: **MATD3-BASE-SH-NOCHUNK-DEFAULT-S44-971f7cc2** paired with **MASAC-BASE-SH-NOCHUNK-MASKAWARE-S44-f6a196a7**, explicitly **no PQL** and no Kaiming. Completed and verified 08/11/2026 02:16:55 UTC+7. Code + 9 PNGs + README commit: `2dd9229`.

- Read exact 20,000 episode History records per run from live W&B, same world/reward/sensor/network/energy, 6UAV, 20 targets, 15 obstacles, 1km, 1500 horizon, seed44, 4096 envs. Not the 35t25o new config. Differing MATD3 default init vs MASAC mask-aware means this is a selected-run exploratory graph, not single-factor controlled comparison.
- Added `scripts/plot_latest_base_small_reward_windows.py`, a four-window **moving average** (8/16/32/64, no downsampling) and **non-overlapping block means** (8/16/32/64), zoom at 4096 boundary. Read-only W&B, no PQL or validation used. Images in `docs/research/reward-small-window-selected-base/`, raw records + analysis.json in local ignored `outputs/reward-small-window-default-matd3-maskaware-masac/`.
- Tests: 32 passed, no failure (7s); Git diff --check clean. Δ reward around 4096 (mean of 64 raw completed episodes each side): MATD3 +1121.64, MASAC -425.32. This persists at even 8-window, so the 200/256 smoothing choice did **not cause** the episode cohort discontinuity; using smaller windows resolves sampling/detail but cannot remove real logged jumps. It remains uncertain what fraction is episode completion-order vs actual policy learning.
- **Kept previous checkpoint save interval 256 episodes** untouched; the '256' checkpoint interval is completely separate from the graphical aggregation windows. Also no W&B Workspace mutations, checkpoint selection/validation adjustments, Kaggle run, algorithm training modifications, or PQL modifications.
- Next: user to inspect these figures and decide display window or request an alternate validation-Return convergence chart. Do not switch Validation schedule without permission. Github branch `experiment/hfbest-embedded-source-steps1500-20261105`.

---

# 2026-11-08 — Plot training Reward from actual W&B History using source-backed methods

Task: user explicitly requested making Reward learning-curve figures from Training History (MADDPG-style block means / EPyMARL-style rolling mean) first; user will decide later if switching the paper plot to periodic fixed-policy validation reward. Started ~08/11/2026 01:54 UTC+7; verified 08/11/2026 02:02:20 UTC+7. Main implementation + PNG commit: `db2ff5b`.

- New script `scripts/plot_training_reward_from_wandb.py` and four pure regression tests. Read-only W&B History, no metric update, no job submission, no training/validation/reward/CBF/observation/optimizer changes. Tests: **28 passed in 6.79s**, 0 errors; git diff --check clean.
- Generated and committed three original analysis PNGs plus README under `docs/research/reward-convergence-preview/` and cached full source records/JSON locally under `outputs/reward-method-study/` (ignored by git).
- Four verified W&B runs: MATD3 BASE default `971f7cc2` (20k), MATD3 PQL `30bdc182` (20k), MASAC BASE `62cfc303` (20k), MASAC PQL `43494a2c` (**16,384 incomplete**). All their W&B configs **match world/sensor/reward/energy/network**, actual scenario 6 UAV, 20 targets, 15 obstacles, 1km, horizon 1500, seed44; they are not new 35t25o. Never mistake plots for final benchmark.
- Figure 1: mean Reward over each complete 256-episode block (MADDPG-inspired, final <256 partial window omitted). Figure 2: sliding 200-completed-episode mean (exact formula from official EPyMARL `plot_results.py::smooth_data`, display only downsampled). Figure 3: per-0.25M-global-step completed-episode-bin means with sparse bins blank (diagnostic, not code taken from a paper).
- All 3 reveal reward/episode-cohort artifacts: abrupt steps at multiples of **4096 episodes**, unchanged materially by smoothing; global-step view has disconnected traces from sparse completion intervals. MATD3 BASE start/end complete block mean -622/+118; MATD3 PQL -3360/-125; MASAC BASE -478/-3467; MASAC PQL partial -401/-3290. Do NOT claim this proves all reward differences artificial; real policy updates occur between cohorts, and some algorithms have poor training reward.
- Current main W&B Workspace is **not modified**; original 25 plots unaffected. Validation still 3,686,400 steps, 5 seeds, best chosen from held-out validation; checkpoint 256 per prior commit. No Kaggle old notebooks regenerated. Next task: user decides whether paper uses these logged training-return curves or requests more-frequent fixed-seed validation reward; do not modify validation until user instructs. Re-render figures using latest run IDs with script options `--run LABEL=RUN_ID`.

---

# 2026-11-08 — Active checkpoint save cadence 128 → 256

Task start: 08/11/2026 01:30:30 UTC+7 | Verified: 08/11/2026 01:31:30 UTC+7. Main code commit: `27bc84d`.

User explicitly requested changing checkpoint saves to 256 completed episodes, then continuing discussion of validation frequency (do not change validation yet). Updated 3 **current 20k/41400s** YAML profiles: comparison-reference-masac-matd3-xyz-sensing, comparison-independent-baseline-xyz-sensing, comparison-independent-35t25o-h512-xyz-sensing. Added targeted config regression test `tests/test_active_checkpoint_cadence.py`. Targeted suite: 94 passed, 2 skipped, 608 subtests passed in 19.02s; git diff --check clean. No algorithms, reward, learning, validation, checkpoint retention (keep=2), or final saving touched. Old/specialized YAMLs and existing embedded Kaggle UI notebooks deliberately unchanged. Changes affect only future runs using latest source/config.

Validation remains every 3,686,400 global transitions with 5 independent fixed-seed CPU episodes and single active evaluator; busy/timeout can skip scheduled points. Source step snapshots are separate from 256-episode regular checkpoints; `best.pt` is validation-selected, `final.pt` is saved on normal exit. User is considering a variable cadence tied to 20k episodes (e.g. 10 milestones); no validation schedule or changes were authorized. Future requirement: study an episode-derived validation schedule robust to bursts from 4096 parallel environments and CPU evaluator busy before coding.

---

# 2026-11-08 — Training and held-out evaluation trade-off metrics

User requested that 4 supplemental plots appear in BOTH Training Convergence and Periodic Evaluation, and asked whether training metrics alone should select best checkpoint before a harder-seed final test.

Start: 08/11/2026 00:21:52 UTC+7 | End (implementation verified): 08/11/2026 00:26:50 UTC+7 | Code commit: `609cc2c`.

- Added passive per-completed-training-episode scalars under `training/*`: `delivery_given_discovery_fraction`, `first_discovery_to_gcs_delay_delivered_mean_s`, `false_confirmation_per_1000_measurements`, `map_coverage_percent`. No learning/reward/CBF/checkpoint-selection changes; missing denominators remain null (not fabricated as zero).
- Moved all 4 existing `evaluation/*` plots into fixed-seed Periodic Evaluation and placed 4 training counterparts in Training Convergence. Removed only redundant Supplemental SECTION, not history. Current dashboard: 9 training + 9 periodic evaluation + 7 collapsed learning diagnostics = 25 charts. Remote W&B Workspace read-back verified the exact 8 metrics and episode X axes scaled by 10.
- Tests: 71 passed, 594 subtests passed, 7 third-party warnings (plus CPU/Warp parity smoke). No Kaggle notebooks regenerated, no jobs submitted, no historical metrics backfilled. Old training runs may have empty `training/*` lines until future runs log them.
- Method recommendation: training episode metrics monitor learning only; select best on disjoint held-out validation seeds (same scenario parameters, new layouts), then one in-distribution final test and optional harder OOD stress tests. Final-test data never selects the checkpoint. Existing code still uses 5 periodic validation seeds at 3,686,400-transition spacing and lexicographic success/delivery/search/reward selection; no protocol change was requested or applied.
- Five existing MATD3 UI notebooks are still older embedded source/timeout than repo. Do not claim they include this metrics update until explicitly rebuilt and verified; do not automatically rebuild or submit them.

---

# 2026-11-07 — Timed handoff after later metrics and budget commits

User requests a paste-ready new-chat prompt including work times. Read
current source and five Library notebook headers before writing the handoff.
Runtime snapshot isfa5ee6507e6c6a78c9258b7ac46e5d3cf9faac4c,23:40:32+07.
The earlier sections below remain historical records, not latest runtime.

Verified post-export history (all07Nov,+07):9243c40 21:26:59 adds five
core evaluation metrics;8efcdb2 21:33:15 changes episode display to tens;
2853c9d 22:13:25 revises train/eval views and compresses journals;
e19c14c 22:40:45 restores compact diagnostics;37eb302 23:19:26 adds coverage
and search/relay trade-offs;fa5ee65 23:40:32 changes three YAML train limits
36000->41400. No learner equations/reward/CBF changes in these commits.

Version mismatch is CONFIRMED and must not be silently described as done:
- Library five-case notebook headers still cite0ca4f76 and10h.
- Downloads five-case files now embed9243c40696188e9faf6d6f4d9aed29517fb23257;
  archived YAML still36000s; file mtimes21:27:14+07.
- Current three active YAMLs have41400s. Builder specs still39600s and
  cloud/remote.py clamps available time against39600 before overhead reserve.
Do not claim full11h30 end-to-end notebook execution is already implemented.

Current evaluation still uses5 fixed validation layouts every3686400 steps
for best selection. Final-only evaluation was discussed, not applied;
training_only notebooks skip auto final30. Best criteria remain success,
delivery,search,return (lexicographic). New core metrics are evaluation-only.
Current chart definitions separate train,held-out checkpoint curves,
search/relay trade-offs and diagnostics. Online view state not reread here.

Fresh targeted checks onfa5ee65:58 passed,7 third-party warnings in4.02s
(core mission/comparison metrics,dashboard,axes,journal,reference contract).
The earlier1374-pass full suite in309.13s belongs to0ca4f76 and is not a
whole-suite verification of the later runtime. No GPU job was submitted or
live training status claimed in this handoff. Historical quota is not fresh.

NEXT: deliver PROMPT_UAV_NETWORK_2026-11-07_CO_MOC_THOI_GIAN.md with exact
commit chronology,configuration,artifact identities and pending work.
Future actions should record actual start/end/duration and source/run IDs.
This task changes continuity notes only; no notebook/runtime regeneration.

---

# 2026-11-07 — Five MATD3 UI ablation notebooks

User requests only manual Kaggle UI files, no MASAC or GPU submission.
Matrix on common35t25o/6UAV/1km/1500step/seed44 scenario:
01 SH,H512,U8,actorLN=true;02 SH,H512,U16,actorLN=true;
03 IND,H512,U16,actorLN=true;04 IND,H512,U16,actorLN=false;
05 SH,H256,U8,actorLN=true. Both actor AND critic sharing/width change;
LayerNorm uses the existing actor-only option; critics stay plain.

Started clean at6c5cb7b. New five-case builder composes the existing kernel
bootstrap and scaled profile with explicit active MATD3 overrides. Original
ui_notebooks.py PREFLIGHT has only a generic status/comment text change.
No algorithm,scenario,observation,CBF,reward,energy,init or noise changes.
Old notebooks/profiles are controls;8 existing local copies were hashed.
All5 new files share one cohort group, explicit SH/IND/ALN/H/U/seed labels,
unique per-execution run IDs, no warm-start, no auto-continue, training-only.

Observed8 RED missing-builder failures before implementation. Fresh related
suite89 passed,590 subtests passed, including actual requested architecture,
CPU Warp transitions/replay,finite U8/U16 updates and delayed4/8 actor calls
for all5 cases. Generated SPEC/config/name propagation checked atseeds44/45.
Fresh full repository suite PASSED:1374 passed,103 skipped,8 third-party
warnings,1662 subtests passed in309.13s; actual exit0.
Read-only review approved: no Critical/Important/Minor findings.
No GPU job was submitted.

Spec/plan:superpowers specs/plans2026-11-07-matd3-five-ui-notebooks.md.
Guide:../../methods/MATD3_FIVE_KAGGLE_UI_ABLATIONS.md.
Implementation commit0ca4f76837a20195b3c06f0279561aaec5ff0758 was pushed.
Exported exactly5 new MATD3 files from that clean source into Downloads
and outputs/kaggle. Actual notebook schemas/code,seeds44/45 resolved
configs,names,tags and all421 source archive files verified. All5 use
archive SHA256cd9ac15600e344b7949d6849bb0a9aca7dfbfb6af40be4d404a7ad95cf02b376.
Eight old notebook copies retain exact checksums. Five user-facing notebooks
were saved separately and byte-verified. No MASAC file was exported.
Evidence:2026-11-07-matd3-five-ui-notebook-verification.json and
2026-11-07-matd3-five-ui-test-results.json in evidence/.
NEXT: user imports the five Private notebooks in Kaggle UI,uses T4x2,Internet,
Secrets and real NOTEBOOK_REF,then fresh Save & Run All. Do not submit jobs
automatically. Compare01vs02,02vs03,03vs04,01vs05 at equal transitions.
T4 batch8192 memory,learning performance and convergence remain unverified.

---

# 2026-11-07 — Actor privacy audit and requested credential maintenance

User asked whether actor198->228 violates Dec-POMDP, and requested checking,
pruning and formatting newly appended Kaggle keys. Started clean atf7c0133.
No runtime algorithm, observation, CBF, reward or notebook source was changed.

Verified observation_dim=158+2*num_targets: only own known/report-status
vectors add30 fields for20->35 targets. Obstacles15->25 alone add ZERO actor
fields. Actor uses3 sensor-footprint obstacle slots; privileged full geometry
is reserved for critic state and the separate CBF layer, not actor tensors.
Local belief/coarse memory and ideal in-range radio/admission metadata are
part of the declared observation model. These assumptions must be stated.

12 counterfactual/positive-control probes passed over Native and Warp CPU:
hidden target/obstacle geometry, other-agent private known data and an
unannounced GCS receipt leave observer inputs and deterministic MASAC/MATD3
raw actions unchanged; critic state changes. Own acquired knowledge or a
visible obstacle changes actor observation as expected. Related regression
suite:59 passed,3 CUDA-hardware tests skipped. No GPU job submitted/needed.
Finite probes do not establish a formal all-rollout privacy theorem.

Important deployment qualification: current CBF reads all-agent geometry
and all obstacles. This does not feed global state into the actor or by
itself invalidate a Dec-POMDP transition model. Do not describe the ENTIRE
safety controller as local-only decentralized execution. Current truthful
claim: CTDE policies with a safety filter using privileged global geometry.
See ../../methods/DEC_POMDP_OBSERVATION_AUDIT_2026-11-07.md and the two
2026-11-07-decpomdp-observation-* evidence files, including repro probe code.

Credential maintenance was local/private, outside this repository. Parsed
6 existing accounts+11 appended tokens. The default CLI was too old to use
KAGGLE_API_TOKEN; its failed probes were NOT used for removal. Rechecked
with the installed automation environment's Kaggle2.2.4:13 accounts returned
GPU quota30h each (2 old+all11 new);4 old entries still returned no quota
after2 attempts. Removed those4 from the active JSON and renamed retained
entries account_1..account_13. Verified original private backup, source-hash
concurrency guard, candidate token set, JSON/load_credentials,0600 mode and
unchanged W&B/HF/non-account fields before/after atomic replacement.
No tokens or credential files were copied to Git, Library or notebook.

Audit/evidence commit d2ee64348831edb2f452c2926290bda8818c4c0f is the verified report.
Final structure check:2 passed,585 subtests passed.
NEXT: answer actor-dimension concern with the exact local-memory breakdown,
mention CBF deployment qualification and completed credential cleanup.
Do not change safety/observation assumptions or launch training based only
on this question. Existing20t15o/256 and35t25o/512 notebooks remain unchanged.

---

# 2026-11-07 — Separate35t25o/512x512 independent experiment

User asks whether16/32 updates help and explicitly requests wider networks
and35 targets/25 obstacles. Updates remain8: the16/32 question is advisory.
Starting clean source ee6a097. Created a separate scaled YAML and --scaled
notebook export option; default profile/files remain the20t15o/256 controls.
No algorithm/reward/CBF/battery, learning rate, seed or sensor-slot changes.

Actual dimensions: independent actor198->228, state565->830,
critic including joint actions631->896. Historical SH actor input was204
(198 observation+6 agent-ID). Online parameters grow about3.18x for both
algorithms; this combines scene and width changes, not a width-only effect.
At batch8192/env4096, U8/U16/U32 consume16/32/64 replay samples per new
joint transition. These are NOT literal per-transition optimizer counts.

RED:2 missing-scaled-API failures observed. GREEN:34 relevant tests and
582 subtests passed, including real Warp CPU transitions,512-network
critic+actor updates and finite metrics for both algorithms. The initial
new probe used Action instead of vector dict; corrected the probe adapter,
not production.36 scenarios generated successfully on first attempt with
no fallback. Baseline notebook hashes remain exactly unchanged.
Fresh full repository suite PASSED:1361 passed,103 skipped,8 third-party
warnings,1654 subtests passed in293.69s; actual exit0.
Read-only review approved with no Critical/Important/Minor findings.
No GPU training was submitted;
T4 batch8192 fit and convergence remain unverified.

Evidence: evidence/2026-11-07-scaled-independent-preflight.json.
Methods: ../../methods/SCALED_INDEPENDENT_BASELINE_35T25O_H512.md.
Implementation source89f3301a976d0f85478d6ce4fce8ce076e06b2ca was pushed.
Exported only labelled35T25O_H512_U8 MASAC/MATD3 notebooks from that clean
source. Both copies, actual schemas/cell compilation, all419 archive files,
seed44/45 launch configs and control hashes verified. Archive SHA256:
6c7516078ef75c76cefc36e80a352e49d48df445332beda0e8d5a80c6439efa8.
New matched notebooks were saved separately; old controls were not replaced.
Export manifest:evidence/2026-11-07-scaled-ui-notebook-verification.json.
Full suite:evidence/2026-11-07-scaled-independent-test-results.json.
NEXT: user may run the new U8 pilot in fresh Kaggle sessions. Keep equal
transition budgets; test16 separately on the same scene/width before32.
Do not claim512/scaling improves learning before GPU evaluation, and do
not launch a job or alter update count without the user selecting that action.

---

# 2026-11-07 — Independent baseline Kaggle UI notebooks

User requested continuing the source-baseline work. Prepared two separate
manual Kaggle notebooks for the existing independent profile: MASAC and MATD3,
BASE, NOCHUNK, plain MLPs, seed44. Scientific YAML/algorithm/reward/CBF unchanged.
The common bootstrap can consume an explicit frozen archive and Kaggle
Secrets; legacy private-dataset credentials remain default. The repository
builder needs no old notebook template from Downloads. Run name/tags track
edited seed and algorithm; each launch is fresh.

Read AGENTS.md and previous task record; initial clean source was
3c46ce91dcac1c358c49bc81479c14582bbdbc9d on the existing experiment branch.
Observed11 RED failures, then79 focused tests and6 subtests passed, including
the actual source-profile BASE/PQL CPU collector/reload checks.

The sandbox tunnel temporarily disconnected before the first full-suite
command was observed to launch. During that outage, fetched415 main/upstream
runtime files from exact GitHub revisions and verified every Git blob SHA.
Ran the11 notebook-boundary tests independently in the artifact runtime.
Exported provisional files from published3c46ce9 runtime and saved them with
a recovery patch. GitHub API write returned403, so no branch/commit was
created through that connector. Sandbox later recovered. SHA256 checks
confirmed all4 edited Python files were BYTE-IDENTICAL across both machines.

The initial full run exposed historical MASAC best-state identity failures:
metadata could retain explicit temperature_loss=log_alpha while canonical
identity omits that equivalent default. Restore accepts both forms while
still rejecting changed alpha objectives, architecture, config or provenance.
The historical actor-normalization matrix includes the changed-objective guard.

Final broad verification on recovered source: 1356 passed, 103 skipped,
8 third-party warnings,1644 subtests passed in 634.26s. The 4 reported failures
were static guards in tests/test_project_structure.py: three >500-line source
subtests and one stale algorithm-hash test. Refreshed only three reviewed pins
for earlier user-authorized 9cd3a15/458d7fb/2e50114 changes; optimization/action
utils pins remain unchanged. Removed blank lines only from configuration.py
and tracking.py, with exact AST-equivalence hashes recorded. Older notes were
archived byte-for-byte below 500 lines/file. Guard+notebook rerun: 13 passed,
578 subtests passed. The broad run is retained with its original exit1; do not
describe it as a fresh whole-suite exit0 after this format-only guard refresh.
No known failures remain after the targeted guard rerun.

Independent read-only review: no Critical/Important findings. Its minor
fresh-session guidance was added to notebook and guide. Same-session rerun
support, T4 memory and convergence are not claimed. No GPU job was submitted.

Implementation source commit 789bc1a0f0976922e351f59fe2b72a32d0e9a368 was pushed
to experiment/hfbest-embedded-source-steps1500-20261105. Built only the TWO
new independent BASE notebooks from that clean source; old SH/init notebooks
were not modified. Downloads and outputs/kaggle copies match. Actual notebook
schema/code compilation, 417 archived source files against current bytes,
same archive in both notebooks, scenario equality, default MATD3 init,
MASAC alpha-dual/MASKAWARE, independent plain MLPs, and seed 44/45 config/name
propagation were verified. Archive SHA256:
47ce6d1bc875b932cdfd03976f3a4f9d9d629a886623211e91528470d5764ee7.
Manifest: evidence/2026-11-07-independent-ui-notebook-verification.json.

Final downloadable notebooks, patch and verification manifest have replaced
all provisional versions successfully; local checksums match the Downloads
copies. Verification notes commit 1df65401971d18cb335389fa13b3884903badc67 was
pushed. No GPU job was submitted. The remaining handoff is manual Kaggle UI:
separate Private notebooks, T4x2 and Internet, WANDB_API_KEY/HF_TOKEN Secrets,
real NOTEBOOK_REF, seed 44, fresh Kaggle execution via Save & Run All.

NEXT: user runs the new matched pair; inspect the resulting real GPU logs and
checkpoint metrics before claiming memory fit, convergence or improved MASAC
ranking. Preserve old SH runs as controls; compare equal transition budgets.

---

# 2026-11-07 — Main MASAC/MATD3 pair audited before initialization ablations

User explicitly prioritized MATD3 `971f7cc2` and MASAC `f6a196a7`; both use
the default initialization scheme. Starting clean source was
`e29e729d5c12270df3417e3d338237c54c82b7ac` on
`experiment/hfbest-embedded-source-steps1500-20261105`.

Verified online W&B finished status, full episode/learner/alpha histories,
raw logs, saved config and exact validation ledgers. At matched18.432M steps
on the same5 validation layouts: delivery63% vs47%, return181.345 vs-331.266.
SHA-verified the exact18.432M historical actors from HF revision history:
MATD3 `a1ffcfcc377d0276ff7fe4292e87b76a5735ab17`,
MASAC `7f6f98935f7a6b947e8b433996aa5a0a3632e1ff`.
Ran30 NEW paired CPU layouts, seeds2110000044..2110000073, plus30
seeded sampled-policy MASAC diagnostic episodes. Holdout MATD3 vs MASAC:
delivery65% vs54.5%, return20.769 vs-286.855. Paired scenario-bootstrap
MASAC-minus-MATD3 delivery=-10.5pp,95%CI[-16,-5.333]; return=-307.624,
CI[-502.912,-101.134]. These are conditional-on-policy scenario intervals,
not training-seed uncertainty. Both0/30 full20-target missions; observed
post-filter safety violations0. MASAC sampling did not resolve the gap.
All-UAV-battery termination already works:29/30 MATD3 and30/30 MASAC mean
episodes ended on all-UAV depletion before the1500-step cap.

The main run's temperatures are adaptive and bounded in observed history,
not the old2734dbf3 explosion. Observed extra safety cost plus lower mission
score explain the return deficit arithmetically. No unique causal training
defect is proven from one training seed; do not force MASAC's ranking or
silently change reward/CBF to make curves agree.

Implemented opt-in `masac.temperature_loss=alpha`, matching the SAC author's
alpha-weighted dual objective. Default `log_alpha` and old checkpoint
selection identities remain compatible. Added a separate
`configs/comparison-independent-baseline-xyz-sensing.yaml` for independent
plain-MLP actors/critics in BOTH algorithms, same scene/reward/CBF/battery,
default MATD3 init, MASKAWARE hybrid interface, original alpha objective.
This is a declared adapted recipe, not literal whole-author-code reproduction
or a verified performance fix. Existing SH profile and historical runs stay
reproducible. No new GPU training, job submission or convergence test.

Fresh focused suite:139 passed,8 skipped,3 subtests passed in82.72s.
Additional old-vs-new temperature checkpoint-identity guard:1 passed.
Code provenance and complete findings:
`docs/methods/MASAC_MATD3_MAIN_PAIR_AUDIT_2026-11-07.md`.
Per-scenario numeric evidence and unsmoothed comparison chart are in
`docs/superpowers/evidence/2026-11-07-*`. Primary author sources are pinned
in the methods report; OpenAI itself has TD3>SAC benchmark cases.

Main-pair implementation/evidence commit
`9cd3a1553639a5928df68a4c91104d86e026a08c` was pushed to the working branch.
Secondary work, after the main audit: corrected cd07de3c's W&B display name
from DEFAULT to KAIMING; readback config/summary hashes were unchanged.
Fixed the notebook generator's DEFAULT-for-KAIMING label mapping and tested
all three generated run names. Labeling implementation commit:
`a2adb5d605005f42ca1c52ed908605eae9ba0667`. Fresh initialization/header suite:
11 passed. Regenerated ONLY the manual Kaiming notebook in Downloads and
outputs/kaggle, from that clean committed source; no Kaggle job submission.
Embedded archive SHA256
`632643cb1e0c6f91381121c79bdf91ea27a41e8feda6ed80d3a228137e31e72a`,
415 files. Notebook cells compile; nbformat validates; actual generated
header resolves KAIMING; archived runtime sources match; both copies match.
Verification manifest: evidence/2026-11-07-kaiming-notebook-verification.json.

NEXT: explain the measured main-pair gap and causal limits to the user.
Use source recipe only as a new labelled baseline; do not claim improved
learning before full GPU training or launch training without a new request.
Preserve old SH runs as controls. A paper comparison needs several training
seeds and matched transition checkpoints.
Do not launch production training automatically. If the new source profile
is selected for a paper, BOTH algorithms need fresh training lineages and
multiple training seeds with matched transition checkpoints.

---

Historical notes preserved byte-for-byte in order: [Part1](CURRENT_TASK_HISTORY_2026-11-07_1.md) | [Part2](CURRENT_TASK_HISTORY_2026-11-07_2.md).
