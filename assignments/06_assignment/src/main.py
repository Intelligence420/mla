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
    # Pipeline:
    #   1. fuse(s, p) -> sp (single K-dim, |sp| = 4096)
    #   2. split sp / x / y -> (seq, prim) pairs with prim sizes 32, 64, 64
    #   3. permute into Slide-5 L2 pattern: [PAR..., k_seq, prim_m, prim_n, prim_k]
    #   4. make_executable() sets exec_types + verifies
    opt = Optimizer(cfg)
    opt.fuse_dims(2, 3)        # s,p -> sp           [a, c, sp, x, b, y]
    opt.split_dim(2, 128, 32)  # sp  -> sp_seq, sp_prim
    opt.split_dim(4, 20, 64)   # x   -> x_seq,  x_prim
    opt.split_dim(7, 24, 64)   # y   -> y_seq,  y_prim
    # current order: [a, c, sp_seq, sp_prim, x_seq, x_prim, b, y_seq, y_prim]
    #                 0  1    2        3       4       5    6    7       8
    # target:        [a, c, x_seq, b, y_seq, sp_seq, x_prim, y_prim, sp_prim]
    opt.permute_dims([0, 1, 4, 6, 7, 2, 5, 8, 3])
    opt.make_executable()

    opt_labels = ['a','c','x_seq','b','y_seq','sp_seq','x_prim','y_prim','sp_prim']
    print("\nTask 3 — optimized Config:")
    print(pretty(cfg, opt_labels))

    print( "Finished." )
