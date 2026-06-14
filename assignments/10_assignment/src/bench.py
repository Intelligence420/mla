"""Messung: mittlere Kernel-Zeit und abgeleitete FLOP/s (whole NPU)."""
import time
import numpy as np
import torch
import pyxrt

insts = np.fromfile("build/insts_matmul.bin", dtype=np.uint32)
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("build/final_matmul.xclbin")
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
context = pyxrt.hw_context(device, uuid)
kname = xclbin.get_kernels()[0].get_name()
kernel = pyxrt.kernel(context, kname)
bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_instr.write(insts.tobytes(), 0)
bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

torch.manual_seed(42)
i0 = torch.randn(256, 1024, dtype=torch.bfloat16)
i1 = torch.randn(1024, 128, dtype=torch.bfloat16)
o = torch.zeros(256, 128, dtype=torch.bfloat16)
b0 = pyxrt.bo(device, i0.nbytes, pyxrt.bo.host_only, 0)
b1 = pyxrt.bo(device, i1.nbytes, pyxrt.bo.host_only, 0)
b2 = pyxrt.bo(device, o.nbytes, pyxrt.bo.host_only, 0)
b0.write(i0.view(torch.int16).numpy().tobytes(), 0)
b1.write(i1.view(torch.int16).numpy().tobytes(), 0)
b2.write(o.view(torch.int16).numpy().tobytes(), 0)
b0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, i0.nbytes, 0)
b1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, i1.nbytes, 0)
b2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, o.nbytes, 0)

for _ in range(10):
    kernel(3, bo_instr, insts.nbytes, b0, b1, b2).wait()
N = 200
t0 = time.perf_counter()
for _ in range(N):
    kernel(3, bo_instr, insts.nbytes, b0, b1, b2).wait()
t = (time.perf_counter() - t0) / N
flop = 2 * 256 * 128 * 1024
print(f"mean time: {t*1e3:.3f} ms")
print(f"throughput: {flop/t/1e9:.1f} GFLOP/s")
