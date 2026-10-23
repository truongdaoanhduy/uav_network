"""CUDA exactness gate for the experimental framefree scheduler."""
import unittest

import torch

from test_frame_free import (FrameFreePrimitiveTests, FrameFreeEmitTests,
                             FrameFreeEventTests, FrameFreeVectorTests,
                             FrameFreeStreamingParityTests)

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaPrimitives(FrameFreePrimitiveTests):
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaEmit(FrameFreeEmitTests):
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaEvents(FrameFreeEventTests):
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaVector(FrameFreeVectorTests):
    device = "cuda"


@unittest.skipUnless(CUDA, "CUDA device required")
class FrameFreeCudaStreaming(FrameFreeStreamingParityTests):
    device = "cuda"
