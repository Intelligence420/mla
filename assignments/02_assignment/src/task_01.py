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
    print(f"Einheiten sind: L2CacheSize, MaxSharedMemoryPerMultiprocessor → Bytes, ClockRate → kHz")
    print(f"========================================")

    # aus gründen, kann .items nicht ordnetlich mit argumenten gefüttert werden
    # sodass es immer alle: (key, value)-Paare --> https://docs.python.org/3/library/stdtypes.html#dict.items
    # zurückgibt. Also muss ich danach filtern und dann printen

if __name__ == "__main__":
    report_device_properties()
