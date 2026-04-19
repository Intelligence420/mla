"""
Task 1: GPU Device Properties

Report L2CacheSize, MaxSharedMemoryPerMultiprocessor and ClockRate
on the DGX Spark using CuPy.
"""

import cuda.tile as ct
import cupy as cp


def report_device_properties():
    """Report specific GPU device attributes."""

    print(f"========================================")
    print(f"Device Name: {cp.cuda.get_device_name()}")
    print(f"")
    print(f"L2CacheSize: {cp.cuda.Device().attributes.items('L2CacheSize')} Bytes")
    print(f"MaxSharedMemoryPerMultiprocessor: {cp.cuda.Device().attributes.items('MaxSharedMemoryPerMultiprocessor')} Bytes")
    print(f"ClockRate: {cp.cuda.Device().attributes.items('ClockRate')} kHz")
    print(f"========================================")

if __name__ == "__main__":
    report_device_properties()
