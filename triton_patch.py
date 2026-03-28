"""
triton_patch.py — Monkey-patch Triton 3.x to skip the CudaUtils compilation step.

When vLLM runs with enforce_eager=True it doesn't use Triton JIT kernels in
the critical hot path. The Triton nvidia driver.py still tries to compile
cuda_utils.c at import time via GCC, which fails on compute nodes because they
lack CUDA development headers (-lcuda resolution fails).

This script must be imported BEFORE any vllm or triton import.
"""

import sys
import types

# Build a mock module for triton.backends.nvidia.driver
# Only stub the classes/functions that break on import.
class _MockCudaUtils:
    """No-op stub — skips the GCC/CUDA compile."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass  # no compilation

    # Stub the three methods vLLM might call via Triton
    def load_binary(self, *args, **kwargs):
        raise NotImplementedError("Triton JIT not available in patched mode")

    def get_device_properties(self, *args, **kwargs):
        return {}

    def cuOccupancyMaxActiveClusters(self, *args, **kwargs):
        return 0

    def set_printf_fifo_size(self, *args, **kwargs):
        pass

    def fill_1d_tma_descriptor(self, *args, **kwargs):
        pass

    def fill_2d_tma_descriptor(self, *args, **kwargs):
        pass


def _patch_triton():
    """Inject the mock into Triton's nvidia driver before it compiles anything."""
    try:
        import triton.backends.nvidia.driver as drv
        # Replace the class on the module in-place
        drv.CudaUtils = _MockCudaUtils
        drv.CudaUtils._instance = _MockCudaUtils()
        print("[triton_patch] Triton CudaUtils patched — GCC compile skipped.")
    except Exception as e:
        print(f"[triton_patch] WARNING: patch failed: {e}")


# Auto-patch on import
_patch_triton()
