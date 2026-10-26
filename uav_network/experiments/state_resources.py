"""Bounded Warp mission-state memory estimates without allocating production env counts."""
from ..environment.gpu.state import allocate_state, allocated_bytes


def estimate_warp_state_bytes(config, env_counts=(1, 64, 512, 4096)):
    counts = [int(value) for value in env_counts]
    if not counts or any(value < 1 for value in counts):
        raise ValueError("env_counts must contain positive integers")
    state, params = allocate_state(config, num_envs=1, seeds=[config.runtime.seed], device="cpu", certify=True)
    shared = sum(value.capacity for name, value in vars(state).items()
                 if name.startswith("fallback_"))
    per_env = int(allocated_bytes(state)) - shared
    return {
        "bytes_per_env": per_env,
        "shared_bytes": shared,
        "estimated_bytes": {str(value): shared + per_env * value for value in counts},
        "fragments_per_report": int(params.fragments),
        "session_capacity_per_env": int(params.session_capacity),
        "packet_capacity_per_env": int(params.packet_capacity),
        # Streaming session state is O(session); packet slots are the only fragment materialization window.
        "session_fragment_dense_bytes": 0,
        "max_queue_packets": int(params.max_queue_packets),
    }
