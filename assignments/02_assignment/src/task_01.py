"""
Task 1: GPU Device Properties

Report L2CacheSize, MaxSharedMemoryPerMultiprocessor and ClockRate
on the DGX Spark using CuPy.
"""

import cupy as cp


def report_device_properties():
    """Report specific GPU device attributes."""

    keys_of_interest = {
        "L2CacheSize",
        "MaxSharedMemoryPerMultiprocessor",
        "ClockRate",
    }

    print(f"========================================")
    for key, value in cp.cuda.Device().attributes.items():
        if key in keys_of_interest:
            print(f"{key}: {value}")
    print(f"========================================")


if __name__ == "__main__":
    report_device_properties()
