# Channel, relay, and energy research assessment

Date: 2026-10-18 (task branch date; inspected during session on 2026-10-17 UTC)
Inspected worktree: `/home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014`
Inspected revision: `61948cd`
Scope: read-only source/code assessment. No training, simulator run, tracked file edit, resource/quota query, or extra agent.

## Recommendation

Keep the existing shared Scenario V2 physical profile for the main MASAC/MATD3 comparison. Its 1 mW–0.4 W logarithmic power action already fixes the demonstrable LOS power-scale problem. The sources do not establish a better numerical urban/suburban coefficient, a need to add fading, or a reason to increase RF battery drain. Describe the simulator as a deterministic, obstacle-conditioned packet-level channel with contention and interference. Qualify its custody/admission assumptions explicitly. Correct/qualify opportunity telemetry before using it as evidence of physically feasible relay opportunities.

Keep all scenario, observation, reward, mask, seed-bank and evaluation changes identical between standard MASAC and MATD3. Do not reward hop count. A different learned relay rate remains an algorithm result.

## 1. What the primary sources actually establish

**UavNetSim-v1.** The authors' paper describes a deliberately simplified packet simulator without encoding/modulation detail. Section III-C uses ACK timeout and retransmissions; section III-F uses transmitted radiation energy `P_tx * t_packet`. Section IV lists communication circuitry, processing, reception and more detailed propulsion as extensions. Its Table I uses fixed noise `4e-11 W`; that is not this project's thermal-noise profile. The current default-branch repository has evolved beyond the 2025 paper, so the vendored revision, not a moving README, defines compatibility. [S1, S2]

**Rotary-wing energy.** Zeng, Xu and Zhang derive a rotary-wing propulsion model, combine propulsion and communication energy, and explain why propulsion generally dominates. They omit extra acceleration energy in their exposition. Their power-speed curve can decrease at lower speeds before increasing; hovering need not minimize power. Consequently the current monotone surrogate `140 + 0.5*v² + 2*a² W` is not their equation. Cite this paper for the qualitative distinction and scale, and label project coefficients as an uncalibrated mission surrogate. Replacing it with a rotor-specific model needs justified rotor parameters and a separate experiment. [S3]

**A2A measurements.** Erdemir et al. measure separate A2A and A2G channels at 3.5 GHz and 100 m UAV altitude in a hilly/forested campus with sparse buildings. Their A2A path loss is relatively close to FSPL, with a fitted exponent 2.166, while A2G behaves differently. This supports distinguishing A2A from A2G, but does not calibrate our 2.4 GHz urban NLOS coefficients or prove that every A2A link is LOS. [S4]

**A2G statistical LOS is a different model.** Al-Hourani's own simulator identifies its A2G sources, computes slant range/elevation between a platform and ground cells, then samples a LOS map from environment statistics. This is useful when geometry is not explicitly known. It is not a parameter source for same-height A2A links; their peer elevation is zero even when their actual unobstructed LOS exists. [S5]

## 2. Current channel: coherent abstraction, limited empirical claim

Pinned upstream `phy/a2a.py` explicitly defines:

| State/model | Path-loss exponent | Fixed excess loss |
| --- | ---: | ---: |
| LOS/free_space | 2.0 | 0 dB |
| NLOS/urban | 3.0 | 12 dB |
| NLOS/suburban | 2.6 | 8 dB |

The 1 m FSPL reference gain is multiplied by distance to the negative exponent and the excess-loss factor. I independently fetched this exact pinned upstream file through the GitHub connector; it matches the checked-in file. The inspected constants/functions have no measurement citation or derivation. It is valid to call them the **pinned UavNetSim analytical urban/suburban profiles**; it is not established that they are measured, 3GPP-calibrated, or representative of this map. [S2]

Project geometry is internally consistent: `communication/upstream.py:scene_payload` creates obstacle prisms of height `sensor.altitude_m+20 = 120 m`; UAVs fly at 100 m and the GCS is on the ground. For these heights, a segment crossing an obstacle footprint intersects the prism. Equal-altitude peer separation is XY distance; GCS path loss uses slant range. This does not claim realistic mixed building heights.

LOS/NLOS and gain are deterministic for fixed cached geometry. Seeded CSMA backoff, motion and overlapping traffic make packet outcomes time-dependent; they do not constitute stochastic fading. A 6 dB threshold applied to overlap-weighted packet SINR is a reception abstraction, not a modulation/coding packet-error curve. No arbitrary Rayleigh/Rician/shadowing coefficient is justified here.

An independent probabilistic LOS override on top of an explicit obstacle map can double-count obstruction or contradict visible geometry. Keep deterministic map-conditioned LOS in core. A future fading or statistical-geometry study should be separately named and calibrated, preserving shared learners.

The 500 m candidate/admission radius is a protocol limit independent of RF reach. Raising power cannot create a destination outside that radius.

## 3. Recomputed link budget

Using the configured `f=2.4 GHz`, `B=22 MHz`, `N0=-174 dBm/Hz`, `NF=7 dB`, `threshold=6 dB` gives noise `4.389577092931546e-13 W`. These numbers below are arithmetic from the inspected implementation/configuration, not empirical measurements.

| XY separation | Peer LOS required power | GCS LOS required power, 100 m height |
| --- | ---: | ---: |
| 100 m | 0.176858 mW | 0.353715 mW |
| 200 m | 0.707430 mW | 0.884288 mW |
| 300 m | 1.591718 mW | 1.768575 mW |
| 500 m | 4.421438 mW | 4.598295 mW |

A 1 mW peer LOS link nominally reaches about 237.8 m; 1 mW no longer reaches every candidate. A 500 m LOS candidate can succeed at roughly 5 mW with no interference, and can fail at the allowed 1 mW. Log mapping gives this region meaningful action resolution. No further numerical change is needed to establish that power controls actual success.

At 500 m, urban NLOS requires 35.0375 W; suburban NLOS requires 1.16130 W. At 0.4 W, their maximum nominal slant ranges are 112.5846 m and 331.8464 m respectively. Thus many blocked urban links are correctly masked by V2 max-power admission. These outcomes follow the selected profile; they are not proofs that real urban radio cannot reach 500 m.

Do not substitute Table I's higher noise without naming a new scenario: it changes the experiment by about 19.6 dB.

## 4. Four distinct communication outcomes

1. **Beyond protocol range:** no admitted link/normal DATA attempt. This is not a weak-signal failure.
2. **Within range, nominal requirement above Pmax:** V2 masks the destination before a normal policy transmission. Count as power-infeasible admission, not attempted DATA failure.
3. **Admitted link, selected power below requirement:** actual emitted DATA can fail the SNR/SINR threshold and incur retries/RF energy.
4. **Sufficient nominal power, concurrent interference or ACK loss:** delivery/MAC outcomes can still fail or require retries. Max-power admission and required-power observation do not promise success.

This distinction is needed when interpreting near-zero weak-signal counts under a strong mask.

## 5. Physical energy and reward interpretation

Current radiation energy uses actual emitted DATA and UAV ACK airtime times selected power, including retries/failed emissions that enter the channel. It excludes RX/listening/circuit energy; GCS energy is outside the UAV battery objective.

For current one-fragment report packetization:
- DATA bits = `8192 + 464 = 8656`.
- DATA airtime = `0.004328 s`.
- One full-power report DATA reference = `0.4 * 0.004328 = 0.0017312 J`.
- A full-power 240-bit UAV ACK = `0.000048 J`.
- A 1 s stationary active-UAV step costs 140 J in the propulsion surrogate, about 80,869 full-power report DATA references.

The hypothetical six-UAV 1800 s stationary propulsion total is 1,512,000 J before battery exhaustion/termination effects. It is not an observed episode result.

The existing normalized RF reward is a defensible **explicit multi-objective preference**: one full-power DATA report-equivalent costs approximately one reward unit, versus +50 on first final delivery. It charges actual RF activity without falsely multiplying physical battery drain. It does not show that communication dominates total physical energy or that coefficient 1.0 is optimal. Report RF joules, propulsion joules, total delivery, delay and RF joules per delivered report separately; interpret a lower ratio together with delivery rate to avoid rewarding silence.

No paper supports inventing PA efficiency/circuit/listening wattages. Add those only from identified hardware/measurements in a separately versioned energy study.

## 6. Relay is optional when carrying is allowed

This is a search-and-store/carry/forward task, not a fixed relay chain. A report bearer outside the GCS radius may move into range and transmit directly before TTL; a relay is useful only when it improves a mission objective relative to carrying, such as delivery delay, continued searching, congestion, or propulsion. Even the farthest unobstructed map point can reduce its horizontal GCS separation to 500 m in about 31 seconds at 20 m/s, far below the 300 s report TTL; obstacles/acceleration/search duties can increase that time. Therefore core settings do not logically require multi-hop on every report.

A successful hop and reduced GCS distance do not alone prove useful routing: the receiver might have no downstream path, be about to expire the report, or displace a better search role. Hop-based positive rewards can produce unnecessary forwarding. Keep delivery outcomes and physical costs as objectives.

Minimal validation:
- Frozen, obstacle-free controlled fixture: source at (500,900), relay at (500,450), GCS at (500,0), novel report at source, other UAVs silent. Direct source-GCS is outside range; both relay legs are within range. At suitable allowed power, require actual two-hop DATA reception and final report delivery, not just selection counts.
- Keep this forced-route fixture separate from the mobile mission. Motion in the mission remains free.
- Measure mission **per-report feasible peer opportunities**, including active peers, range, nominal Pmax feasibility, report-specific receiver admission, buffer and remaining TTL; distinguish direct-GCS-unavailable cases.
- A snapshot feasible path to GCS is stronger evidence than merely a closer peer, but still is not guaranteed future delivery.
- To establish benefit, evaluate matched peer-enabled/peer-disabled policies or oracle baselines under the same mission seeds/budget. Do not claim necessity from learned hop histograms.

### Existing telemetry issue

`communication/gpu/telemetry.py:record_action` computes `ACTION_RECEIVER_ELIGIBLE` from active/range/report-admission conditions, but does **not** apply `power_limited_admission`. Native `backend.py:step` computes eligibility through `_eligible`, which already checks nominal max-power feasibility. Thus the GPU count is a geometric/admission opportunity rather than proof of a V2 nominally feasible channel, and the two backends currently disagree on this telemetry condition. Minimal correction is use the same shared/mirrored power-feasibility predicate, or clearly relabel the count. A direct-vs-peer breakdown is also needed for relay claims.

## 7. Two report-contract assumptions needing explicit wording

### Receiver-side atomic report custody

Native `routing.py:packet_reception` invokes `backend.packet_received` on successful DATA; complete reassembly immediately calls `ReportStore.transfer`, deletes source custody and assigns the peer/GCS. Only afterward does routing wait SIFS and emit ACK. GPU `mac.receive_data` likewise calls `reports.transfer` at complete DATA reception.

Therefore ACK loss can trigger MAC retries/energy without undoing accepted report custody. End-to-end delivery means complete receiver DATA acceptance, not sender receipt of final ACK. This is physically possible as receiver delivery, but globally atomic source-custody removal is an idealization. Do not describe it as ACK-confirmed custody. Reliable metadata/admission handshakes are already explicitly free in `observations.destination_mask`.

The historical experimental Bundle Protocol RFC 5050 distinguishes acceptance from release: section 5.11 releases sender custody after receipt of a success custody signal. It is useful background, not a protocol this project implements. [S6]

Smallest core action: document atomic receiver acceptance and regress ACK-loss-after-successful-DATA behavior in both backends. If ACK-confirmed custody is a real requirement, retain source report pending confirmation, make duplicate/reassembly handling idempotent, and commit source release only when confirmation is decoded. Simply moving transfer into the current ACK handler is insufficient for fragments, lost ACKs and receiver capacity races. Such a change needs focused native/Warp parity validation.

### Permanent target-knowledge admission restriction

Native `ReportStore.can_receive` and GPU `reports.can_receive` require `known[receiver,target] == false` rather than merely no current report copy. Target knowledge survives forwarding and expiry. This suppresses loops, but also prevents a formerly informed peer from receiving a replacement/new report version or recovery copy. Fresh visual rediscovery at a node does not erase another node's permanent knowledge.

Core can retain this if described as **novel target-information dissemination with permanent peer revisit suppression**, not general recoverable packet routing. It still supports a simple multi-hop path through previously uninformed peers.

If the intended contract is ordinary report routing/recovery, minimally separate target knowledge from current report custody and suppress duplicates/loops by report identity (target, origin, creation time) or an explicit per-report visited-node record. Replacing the check with only “buffer lacks target” permits cycles and can invalidate relay-efficiency metrics. Do not silently make this semantic change in one backend or only one learner.

## Sources

[S1] Zihao Zhou et al., UavNetSim-v1 (2025), authors' arXiv paper:
https://arxiv.org/html/2507.09852v1

[S2] Exact pinned upstream implementation:
https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/phy/a2a.py
Fetched successfully via GitHub connector; generic web fetch of this pinned URL failed. Local checked-in source independently read.
Current official repository, context only:
https://github.com/Zihao-Felix-Zhou/UavNetSim
Do not cite moving README features as proof of the pinned backend's behavior.

[S3] Yong Zeng, Jie Xu, Rui Zhang, “Energy Minimization for Wireless Communication With Rotary-Wing UAV,” IEEE TWC 18(4), 2329–2345 (2019), DOI 10.1109/TWC.2019.2902559:
https://arxiv.org/abs/1804.02238
https://arxiv.org/pdf/1804.02238

[S4] Ubeydullah Erdemir et al., “Measurement-based Channel Characterization for A2A and A2G Wireless Drone Communication Systems” (2023), authors' manuscript:
https://arxiv.org/abs/2306.08474
https://arxiv.org/pdf/2306.08474

[S5] Akram Al-Hourani's own NTNChannelSimulator (author implementation and bibliography):
https://github.com/AkramHourani/NTNChannelSimulator
https://github.com/AkramHourani/NTNChannelSimulator/blob/main/A01_Main.m
https://github.com/AkramHourani/NTNChannelSimulator/blob/main/F01_GenLoS.m
Cited A2G paper: A. Al-Hourani, S. Kandeepan, A. Jamalipour, GLOBECOM 2014, DOI 10.1109/GLOCOM.2014.7037248. IEEE page fetched but did not expose useful full text, so no numerical model was borrowed from it.

[S6] RFC Editor, RFC 5050 (historical experimental protocol), sections 5.10–5.12:
https://www.rfc-editor.org/rfc/rfc5050
Applied in this branch: GPU opportunity telemetry now includes the same max-power feasibility predicate as Native, plus exact joint opportunity/silence counters. The assessment below records the pre-fix61948cd evidence.
