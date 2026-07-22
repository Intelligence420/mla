import os

import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt

from config import generate_config, pretty, Config, DimType, ExecType, DataType
from kernel import verify_kernel, benchmark, build_optimized_config

def plot_tensor(
    tensor,
    path='tensor_plot.png',
    title=''
):
    """
    Plots a 5D tensor by slicing along the first two dimensions and displaying the resulting images.
    Dimension order is assumed to be (a, b, c, y, x) where a and b are image indices and c is the color channel.

    Args:
        tensor (torch.Tensor): A 5D tensor of shape (a, b, c, y, x).
        title (str): Title for the plot.
    """
    a, b, c, y, x = tensor.shape
    fig, axes = plt.subplots(a, b, figsize=(b * 2, a * 2))
    for i in range(a):
        for j in range(b):
            img = tensor[i, j].numpy()
            # reorder from c,y,x to y,x,c
            img = np.transpose(img, (1, 2, 0))
            img *= 255.0
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    data = np.load('./data/lf_tr_64_intermediate.npz')
    tensor_acspx = torch.tensor(data['tensor_acspx']).cuda()
    tensor_bspy = torch.tensor(data['tensor_bspy']).cuda()

    # Compute root tensor by calling torch.einsum (FP32)
    tensor_abcyx_fp32 = torch.einsum(
        'acspx,bspy->abcyx',
        tensor_acspx.to(torch.float32),
        tensor_bspy.to(torch.float32),
    )
    plot_tensor(
        tensor_abcyx_fp32.cpu(),
        path='results/torch_32.png',
        title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch FP32'
    )

    # Compute root tensor by calling torch.einsum (FP16)
    tensor_abcyx_fp16 = torch.einsum(
        'acspx,bspy->abcyx',
        tensor_acspx.to(torch.float16),
        tensor_bspy.to(torch.float16),
    )
    plot_tensor(
        tensor_abcyx_fp16.to(torch.float32).cpu(),
        path='results/torch_16.png',
        title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch FP16'
    )

    # Task 2: basic Config via generate_config from Assignment 05
    cfg = generate_config(
        'acspx,bspy->abcyx',
        [tuple(tensor_acspx.shape), tuple(tensor_bspy.shape)],
    )
    dim_labels = list('acspxby')  # erstes Auftreten in acspx, bspy
    print("Task 2 — basic Config (acspx,bspy->abcyx):")
    print(pretty(cfg, dim_labels))

    # Task 3: Optimized Config — der L2-Super-Tile-Swizzle wird DATENGETRIEBEN
    # ueber die Config ausgedrueckt (zwei Split-Ebenen + verschachtelte M-/N-PAR-
    # Achsen), nicht per Hand im Kernel. Siehe build_optimized_config in kernel.py.
    #   1. fuse(s, p) -> sp;  split sp/x/y -> (seq, prim)
    #   2. x_seq/y_seq -> (super, group); Gruppen-Achsen innen verschachteln
    #   3. make_executable() setzt exec_types + verifiziert
    # PRIM-Tiles (64,64,32) wie in Assignment 05 (bester ct.mma-Footprint, FP16).
    cfg_opt = build_optimized_config(
        tuple(tensor_acspx.shape), tuple(tensor_bspy.shape), group=(4, 3))

    opt_labels = ['a', 'c', 'b', 'x_super', 'y_super', 'x_group', 'y_group',
                  'sp_seq', 'x_prim', 'y_prim', 'sp_prim']
    print("\nTask 3 — optimized Config (2D-Super-Tile via Config, GX=4, GY=3):")
    print(pretty(cfg_opt, opt_labels))

    # Task 4: Kernel auf den realen Daten (FP16)
    a16 = tensor_acspx.to(torch.float16)
    b16 = tensor_bspy.to(torch.float16)
    print("\nTask 4b — Kernel-Verifikation gegen torch.einsum:")
    verify_kernel(a16, b16)
    print("\nTask 4c — Benchmark:")
    benchmark(a16, b16)

    print( "Finished." )


"""Ergebnisse
(.venv) mla08@flambe:~/MLA/mla/assignments/06_assignment$ python3 src/main.py
Task 2 - basic Config (acspx,bspy->abcyx):
  [a,c: M | s,p: K | x: M | b,y: N], alle SEQ, data_type=FLOAT16

Task 3 - optimized Config (2D-Super-Tile via Config, GX=4, GY=3):
pos name    type  exec      size
0   a       M     PAR          4
1   c       M     PAR          3
2   b       N     PAR          4
3   x_super M     PAR          6
4   y_super N     PAR          6
5   x_group M     PAR          4
6   y_group N     PAR          3
7   sp_seq  K     SEQ        128
8   x_prim  M     PRIM        64
9   y_prim  N     PRIM        64
10  sp_prim K     PRIM        32
  -> PAR verschachtelt M/N (Gruppen-Achsen innen), PRIM-Block [M,N,K]

Task 4b - Verifikation gegen torch.einsum (echte Daten):
  baseline / generic(ilv) / generic(128) / big-prim  -> alle allclose=True (max_abs_err=0.0010)

Task 4c - Benchmark (FLOPs=6.958e+11):
  torch.einsum (FP16)            12.07 ms   57.67 TFLOPS   1.00x
  baseline 3D (64x64x32)         43.55 ms   15.98 TFLOPS   0.28x
  generic natural (64x64x32)     44.26 ms   15.72 TFLOPS   0.27x   (== baseline, kein Overhead)
  generic natural (128x128x32)   29.40 ms   23.66 TFLOPS   0.41x   (bester Custom, == big-prim)
  big-prim (128x128x32)          29.51 ms   23.58 TFLOPS   0.41x
  generic interleaved (4,3)      57.76 ms   12.05 TFLOPS   0.21x   (Super-Tile hier neutral/langsamer:
  generic interleaved (6,6)      48.86 ms   14.24 TFLOPS   0.25x    B passt pro Batch in den L2)
  generic interleaved (8,9)      46.41 ms   14.99 TFLOPS   0.26x
  generic interleaved (12,9)     46.00 ms   15.13 TFLOPS   0.26x
Finished.
"""
