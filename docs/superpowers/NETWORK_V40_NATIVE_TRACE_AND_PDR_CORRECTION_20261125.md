# Zhang-Xue v4.0: native DATA PDR fix and actual MAC RNG replay

Date: 2026-11-25. Separate research branch:
experiment/zhang-xue-v40-native-mac-trace-parity-20261125.
Parent checkpoint: v3.9 5cf397ce4041c459c72386ebc1240b62b1f65948.
Original classic UavNetSim-v1 pin 64bc3a2424f2976418731ee94f9a8fd3e8810b22 remains unchanged.

## Critical measurement bug, RED then GREEN

In classic_v1_simpy.py, an upstream ApplicationDataPacket correctly
reached its destination and was saved to delivered_packets. But the
local NetworkMetricsFilter.on_deliver() only accepts a local
SearchDataPacket, not upstream ApplicationDataPacket, so DATA PDR
stayed zero despite actual deliveries.

RED test exposed physical delivered IDs [1,2] versus metric [].
Fix explicitly records packet_id in data_delivered_ids in the already
idempotent receiver callback. The previously published v3.9
six-seed native PDR [0,0,0,0,0,0] was WRONG.
Correct native PDR seeds44-49: [0,1,1,0,0,0].

## Differentiate app DATA from reverse link MAC ACK

The Warp v3.9 one-hop prototype incorrectly treated receiving an ACK
at the sender as synonymous with successful application DATA delivery.
DATA can arrive at destination despite its MAC ACK colliding.
The Warp kernel now separately exports:
- data_received: 1 if any DATA frame reaches the app destination;
  duplicate retries do not double count.
- mac_acked: 1 if the reverse MAC ACK reaches sender.
- delivered: legacy alias of mac_acked; NOT the true DATA PDR.
- dropped/expired: sender-side terminal outcomes. A DATA-received
  sender may also later have dropped=1 because the ACK was lost.

The map-to-network smoke now merges remote evidence on data_received
rather than the legacy ACK-confirmed delivered key.

## Verified real native source trace

The new native collision_trace mode captures actual upstream logger
DATA starts/ACK timeout/ACK received/drops plus original per-agent
MAC randint calls without modifying the upstream source or consuming
additional RNG draws. The v4.0 verifier replays those draw values
into Warp and compares true application receiver DATA PDR and times.

One controlled topology only: 0m-400m-800m hidden terminal pair;
two 566-byte DATA senders sharing 802.11b channel 1, 2Mbps and
up to 5 MAC retries. Native packet injection at 25ms; Warp starts
measured relative to injection.

- Seeds44-49: 6/6 true DATA PDR matches.
- Seeds44-75: 32/32 PDR matches, 302 TX attempts each;
  first TX start times exact; 18 later TX starts differ by up
  to 550 microseconds.
- Seeds44-107: 64/64 PDR matches, 601 TX attempts each;
  all first TX starts exact; 35 later TX starts differ by up
  to 550 microseconds. 88 DATA arrived out of 128 generated
  for both native and Warp.
- No unknown native RNG draws were synthesized in these runs.
- Warp CPU only; no CUDA driver on local Ubuntu host.

Repro: set UAV_CLASSIC_V1_ROOT to pinned v1 checkout then run
python -m scripts.verify_zhang_xue_network_v40 --start-seed 44
--num-seeds 64 --output-json path/to/output.json.

Evidence in docs/superpowers/evidence:
zhang_xue_v40_32seed_native_rng_parity_20261125.json
zhang_xue_v40_64seed_native_rng_parity_20261125.json

## Remaining blockers -- full Gate 5 NOT PASSED

This parity is one-hop controlled contention, not general network
fidelity. At seed45 native relay UAV1 additionally sends application
E2E ACK packet 600002 at 48530us. This E2E ACK occupies carrier
resource and shifts source UAV0 final DATA start to 49756us
versus isolated Warp prediction 49206us (550us difference).

Need persistent packet buffers, real app E2E ACKs and watermark,
HELLO discovery and privacy-safe multi-hop routing, CCA backoff
freeze/resume parity, dynamic blackout/TTL and receive fusion
resident on Warp, broader topology / several traffic loads, and
end-to-end E4096 GPU environment test BEFORE any training.

No MATD3/MASAC training, Kaggle GPU job, W&B or HF mutations
were performed. Do not claim global network parity.
