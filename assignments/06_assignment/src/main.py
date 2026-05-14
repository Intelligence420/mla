import os

import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt

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

    print( "Finished." )
