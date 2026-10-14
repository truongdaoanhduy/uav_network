"""CPython integer seeding and MT19937 getrandbits/randint, without Warp RNG."""
import numpy as np
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState

wp.set_module_options({"enable_backward": False})


@wp.func
def seed_mt(s: WorldState, e: int, node: int):
    seed = wp.uint64(s.seeds[e]) + wp.uint64(node + 5)
    if s.seeds[e] < wp.int64(0):
        signed_seed = s.seeds[e]+wp.int64(node+5)
        seed = wp.uint64(signed_seed)
        if signed_seed < wp.int64(0):
            seed = wp.uint64(0)-seed
    key0 = wp.uint32(wp.uint64(seed) & wp.uint64(4294967295))
    key1 = wp.uint32(wp.uint64(seed) >> wp.uint64(32))
    key_length = 1
    if key1 != wp.uint32(0):
        key_length = 2
    s.mt[e,node,0] = wp.uint32(19650218)
    for i in range(1,624):
        x = s.mt[e,node,i-1]
        s.mt[e,node,i] = wp.uint32(1812433253)*(x ^ (x >> wp.uint32(30))) + wp.uint32(i)
    i = int(1)
    j = int(0)
    for k in range(624):
        x = s.mt[e,node,i-1]
        key = key0
        if j == 1:
            key = key1
        s.mt[e,node,i] = (s.mt[e,node,i] ^ ((x ^ (x >> wp.uint32(30)))*wp.uint32(1664525))) + key + wp.uint32(j)
        i += 1
        j += 1
        if i >= 624:
            s.mt[e,node,0] = s.mt[e,node,623]
            i = 1
        if j >= key_length:
            j = 0
    for k in range(623):
        x = s.mt[e,node,i-1]
        s.mt[e,node,i] = (s.mt[e,node,i] ^ ((x ^ (x >> wp.uint32(30)))*wp.uint32(1566083941))) - wp.uint32(i)
        i += 1
        if i >= 624:
            s.mt[e,node,0] = s.mt[e,node,623]
            i = 1
    s.mt[e,node,0] = wp.uint32(-2147483648)
    s.mt_index[e,node] = 624


@wp.func
def mt_word(s: WorldState, e: int, node: int) -> wp.uint32:
    index = s.mt_index[e,node]
    if index >= 624:
        for k in range(624):
            y = (s.mt[e,node,k] & wp.uint32(-2147483648)) | (s.mt[e,node,(k+1)%624] & wp.uint32(0x7fffffff))
            a = wp.uint32(0)
            if (y & wp.uint32(1)) != wp.uint32(0):
                a = wp.uint32(-1727483681)
            s.mt[e,node,k] = s.mt[e,node,(k+397)%624] ^ (y >> wp.uint32(1)) ^ a
        index = 0
    y = s.mt[e,node,index]
    s.mt_index[e,node] = index+1
    y = y ^ (y >> wp.uint32(11))
    y = y ^ ((y << wp.uint32(7)) & wp.uint32(-1658038656))
    y = y ^ ((y << wp.uint32(15)) & wp.uint32(-272236544))
    return y ^ (y >> wp.uint32(18))


@wp.func
def randbelow(s: WorldState, e: int, node: int, bound: int) -> int:
    bits = int(0)
    b = bound
    while b > 0:
        bits += 1
        b = b >> 1
    r = int(mt_word(s,e,node) >> wp.uint32(32-bits))
    while r >= bound:
        r = int(mt_word(s,e,node) >> wp.uint32(32-bits))
    return r


@wp.kernel
def _draw(s: WorldState, p: WorldParams, node: int, bounds: wp.array(dtype=wp.int32),
          output: wp.array(dtype=wp.int32)):
    seed_mt(s,0,node)
    for i in range(bounds.shape[0]):
        output[i] = randbelow(s,0,node,bounds[i])


def draw_backoffs(s, p, bounds, node=0):
    bounds = np.asarray(bounds, dtype=np.int32)
    if np.any(bounds < 1) or np.any(bounds >= 1<<30):
        raise ValueError("randbelow bounds must fit positive int32")
    output = wp.empty(len(bounds), dtype=wp.int32, device=s.seeds.device)
    wp.launch(_draw, dim=1, inputs=[s,p,node,wp.array(bounds,dtype=wp.int32,device=s.seeds.device),output],
              device=s.seeds.device)
    return output.numpy()
