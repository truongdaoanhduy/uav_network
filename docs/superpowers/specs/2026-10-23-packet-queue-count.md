# Packet queue-count optimization spec

Goal: remove the redundant full `packet_capacity` scan used only to count queued packets inside `fill_session()` while preserving exact Warp/native packet semantics.

Constraints:
- No scientific behavior changes: packet order, counts, retries, ACKs, relay, energy, delivery, and timestamps must remain exact.
- Reuse the already-maintained `node_queue_count[e, source]`; add no new state or dependency.
- One optimization only in this task.
- Keep production training untouched; work on isolated branch `perf/packet-scan-20261023`.
- Verify with a structural regression test, GPU/native packet parity tests available locally, and later a T4 benchmark against commit `4cd9358`.
