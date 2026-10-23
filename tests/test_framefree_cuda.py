"""CUDA exactness gate for the experimental framefree scheduler."""
import unittest

import torch

from test_frame_free import (FrameFreePrimitiveTests as _FrameFreePrimitiveTests,
                             FrameFreeEmitTests as _FrameFreeEmitTests,
                             FrameFreeEventTests as _FrameFreeEventTests,
                             FrameFreeVectorTests as _FrameFreeVectorTests,
                             FrameFreeStreamingParityTests as _FrameFreeStreamingParityTests)

CUDA = torch.cuda.is_available()

# Imported unittest bases are helpers here, not separate collected cases.
for _base in (_FrameFreePrimitiveTests, _FrameFreeEmitTests, _FrameFreeEventTests,
              _FrameFreeVectorTests, _FrameFreeStreamingParityTests):
    _base.__test__ = False


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaPrimitives(_FrameFreePrimitiveTests):
    __test__ = True
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaEmit(_FrameFreeEmitTests):
    __test__ = True
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaEvents(_FrameFreeEventTests):
    __test__ = True
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaVector(_FrameFreeVectorTests):
    __test__ = True
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaStreaming(_FrameFreeStreamingParityTests):
    __test__ = True
    device = "cuda"
