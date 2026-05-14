import os

import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt

from config import generate_config, pretty, Config, DimType, ExecType, DataType
from optimizer import Optimizer

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

    # Task 3: Optimized Config
    # PRIM tile sizes follow Assignment 05 (best ct.mma footprint on GB10 / FP16).
    PRIM_M, PRIM_N, PRIM_K = 64, 64, 32
    _, _, s_sz, p_sz, x_sz = tensor_acspx.shape
    _, _, _, y_sz = tensor_bspy.shape
    k_total = s_sz * p_sz
    assert x_sz % PRIM_M == 0, f"x={x_sz} not divisible by PRIM_M={PRIM_M}"
    assert y_sz % PRIM_N == 0, f"y={y_sz} not divisible by PRIM_N={PRIM_N}"
    assert k_total % PRIM_K == 0, f"k={k_total} not divisible by PRIM_K={PRIM_K}"

    # Pipeline:
    #   1. fuse(s, p) -> sp (single K-dim, |sp| = s*p)
    #   2. split sp / x / y -> (seq, prim) pairs
    #   3. permute into Slide-5 L2 pattern: [PAR..., k_seq, prim_m, prim_n, prim_k]
    #   4. make_executable() sets exec_types + verifies
    opt = Optimizer(cfg)
    opt.fuse_dims(2, 3)                           # s,p -> sp     [a, c, sp, x, b, y]
    opt.split_dim(2, k_total // PRIM_K, PRIM_K)   # sp  -> sp_seq, sp_prim
    opt.split_dim(4, x_sz // PRIM_M, PRIM_M)      # x   -> x_seq,  x_prim
    opt.split_dim(7, y_sz // PRIM_N, PRIM_N)      # y   -> y_seq,  y_prim
    # current order: [a, c, sp_seq, sp_prim, x_seq, x_prim, b, y_seq, y_prim]
    #                 0  1    2        3       4       5    6    7       8
    # target:        [a, c, x_seq, b, y_seq, sp_seq, x_prim, y_prim, sp_prim]
    opt.permute_dims([0, 1, 4, 6, 7, 2, 5, 8, 3])
    opt.make_executable()

    opt_labels = ['a','c','x_seq','b','y_seq','sp_seq','x_prim','y_prim','sp_prim']
    print("\nTask 3 — optimized Config:")
    print(pretty(cfg, opt_labels))

    print( "Finished." )
