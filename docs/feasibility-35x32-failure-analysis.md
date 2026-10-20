# Feasibility 35-found / 32-delivered failure analysis

Observed artifact:
`outputs/session-relay-20261020/feasibility-warp-full35-1800/result.json`
(seed `2000000044`, `warp_packet`, 35 targets, 25 obstacles, 1800-step horizon).
This is a privileged engineering oracle, not a learned-policy baseline.

## Observed outcome

- Search: 35/35 targets found (100%).
- GCS delivery: 32/35 reports (91.43%).
- Termination: 1800-step time limit, not success.
- Undelivered target IDs (zero-based): 14, 22, 31.
- Their first discoveries were at 243 s, 232 s, and 344 s respectively; all were
  discovered more than four 300 s report-TTL intervals before mission end.
- Final report custody was empty: 0 reports / 0 bytes in UAV buffers. Diagnostics
  classify all three as discovered-undelivered **without custody**.
- Report lifecycle pressure: 7 expirations, 31 fresh-discovery buffer drops, and
  19 successful report regenerations after later camera re-observation.
- Buffer peaks reached the 3 MB configured limit on five UAVs and 2 MB on the
  sixth.
- Energy was also limiting: five of six UAVs ended at zero battery; the only
  active UAV retained about 3.24% battery.

The packet network was degraded but operational: packet PDR was 89.83%, 32 whole
reports reached GCS (24 direct, 8 relayed), and 17 complete peer-custody transfers
occurred, 16 of them toward GCS. Queue-full, packet-lifetime, and radio-energy
admission counters were zero in this run. Retry-limit and collision/interference
failures were substantial, so network contention increased delay/energy pressure,
but there is no evidence here that the final three reports vanished from a packet
queue bug.

## Root cause classification

The report store deliberately supports rediscovery recovery: an expired or
buffer-rejected report can be recreated only after a fresh camera observation.
Target knowledge remains true after custody is lost so discovery reward is not
repeated.

The privileged feasibility controller currently assigns search work from
`known.any(axis=0)`. Therefore, once a target is known, it is excluded from the
search assignment even if its report has later expired or been dropped and no UAV
holds custody. A focused probe gives:

- undiscovered target -> assignment `[0]`;
- known target with no custody -> assignment `[None]`.

So the controller does not deliberately revisit known-but-undelivered targets to
trigger recovery. Regeneration in the 35x32 run can only happen when an active UAV
happens to observe such a target again while space is available. With 3 MB buffers,
300 s TTL, heavy contention, and five depleted UAVs, three reports ended with no
custodian before the 1800-step deadline.

## Verification

Focused recovery/parity tests executed on the reviewed branch:

- fresh camera observation recreates an expired report;
- buffer drop recovers after space becomes available and re-observation occurs;
- a re-observed report can reach GCS through the real network;
- Warp/native recovery, TTL and receiver-admission state agree in the parity test;
- telemetry correctly reports an expired discovery without custody.

Result: **5 passed**.

## Conclusion

This run does **not** show that Scenario V3 is structurally impossible, and the
specific 35/32 gap is not evidence of a confirmed Warp custody-loss bug. The
observed failure is explained by a feasibility-controller blind spot combined
with finite buffer/TTL, network contention, and battery depletion. This does not
prove the simulator is globally bug-free; it narrows this failure to the tested
lifecycle path.

A follow-up oracle-only improvement may prioritize known-but-undelivered targets
that have no live custody/recovery owner. Such a change must remain isolated from
MASAC/MATD3 baseline learning logic and be regression-tested before rerunning the
feasibility probe.
