"""Debug: Fehler pro (a,x,b,y)-Ausgabeblock aufschluesseln."""
import numpy as np
import torch
import pyxrt

xclbin_path = "build/final_matmul.xclbin"
insts_path = "build/insts_matmul.bin"
insts = np.fromfile(insts_path, dtype=np.uint32)
device = pyxrt.device(0)
xclbin = pyxrt.xclbin(xclbin_path)
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
context = pyxrt.hw_context(device, uuid)
kname = xclbin.get_kernels()[0].get_name()
kernel = pyxrt.kernel(context, kname)
bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_instr.write(insts.tobytes(), 0)
bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

torch.manual_seed(42)
data_in0 = torch.randn(256, 1024, dtype=torch.bfloat16)
data_in1 = torch.randn(1024, 128, dtype=torch.bfloat16)
data_out = torch.zeros(256, 128, dtype=torch.bfloat16)
bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
bo_in1 = pyxrt.bo(device, data_in1.nbytes, pyxrt.bo.host_only, 0)
bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)
bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
bo_in1.write(data_in1.view(torch.int16).numpy().tobytes(), 0)
bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)
to = torch.frombuffer(bo_out.map(), dtype=torch.bfloat16, count=256*128).view(256,128)
bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)
bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)
h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
h.wait()
bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

ref = (data_in0.float() @ data_in1.float())
out = to.float()

# Bloecke: M=a*128+x*16 (a=2,x=8), N=b*64+y*16 (b=2,y=4). Blockgroesse 16x16.
print("Fehler pro (a,x | b,y)-Block (mittlerer abs. Fehler, . = ok <0.5):")
for a in range(2):
  for x in range(8):
    m0 = a*128 + x*16
    row = f"a{a} x{x}: "
    for b in range(2):
      for y in range(4):
        n0 = b*64 + y*16
        blk_o = out[m0:m0+16, n0:n0+16]
        blk_r = ref[m0:m0+16, n0:n0+16]
        e = (blk_o-blk_r).abs().mean().item()
        row += ("  .  " if e < 0.5 else f"{e:5.1f}") + " "
    print(row)
print("Spalten: b0y0 b0y1 b0y2 b0y3  b1y0 b1y1 b1y2 b1y3")
print(f"\nGesamt mean abs err: {(out-ref).abs().mean().item():.3f}")
