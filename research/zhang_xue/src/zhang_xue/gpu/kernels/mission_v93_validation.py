"""Device-side action validation before committing a V80 world transition."""
from zhang_xue.gpu.state_v67 import HAS_WARP, wp

if HAS_WARP:
    @wp.kernel
    def validate_actions_v93(
        actions: wp.array3d(dtype=wp.float32),
        destinations: wp.array2d(dtype=wp.int32),
        failed: wp.array(dtype=wp.uint8),
        safety_reason: wp.array(dtype=wp.uint8),
    ):
        e = wp.tid()
        for u in range(6):
            d = destinations[e, u]
            if (not wp.isfinite(actions[e,u,0]) or not wp.isfinite(actions[e,u,1])
                    or d < 0 or d > 2):
                failed[e] = wp.uint8(1)
                safety_reason[e] = safety_reason[e] | wp.uint8(16)

