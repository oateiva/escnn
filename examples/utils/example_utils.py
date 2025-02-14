import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ---------- Data Handling Functions ---------- #


def read_gabor_data(path):
    """ Reads the Gabor dataset description file. """
    return pd.read_csv(path)


def load_image_from_gabor_data(data_path, gabor_data,
                               orientation=0.0, shift_x=0, shift_y=0):
    """ Loads an image from the Gabor dataset
    based on specified parameters. """
    selected_entry = gabor_data[
        (gabor_data['orientation_0'] == orientation)  # &
        # (gabor_data['shift_x_0'] == shift_x) &
        # (gabor_data['shift_y_0'] == shift_y)
    ]
    image_path = os.path.join(data_path,
                              selected_entry['image_name'].values[0])
    return mpimg.imread(image_path), image_path


# ---------- Visualization Functions ---------- #

def plot_img(img, title='Image'):
    """ Plots a grayscale image with pixel grid. """
    height, width = img.shape[:2]
    plt.imshow(img, cmap='gray', origin='upper', interpolation='nearest')
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    plt.xticks(np.arange(width+1)-.5, labels=np.arange(width+1))
    plt.yticks(np.arange(height+1)-0.5, labels=np.arange(height+1))
    plt.gca().xaxis.set_ticks_position('top')

    plt.title(title)
    plt.gcf().set_size_inches(2, 2)
    plt.show(block=False)


def plot_multiple_3d_tensors(tensors,
                             titles=None, cmap="gist_ncar", title=None,
                             alpha=0.9,
                             aspect_ratio=[1, 1, 1],
                             fig_size=(8, 8)):
    """ Plots multiple 3D tensors side by side. """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors,
                             figsize=(8 * num_tensors, 8),
                             subplot_kw={'projection': '3d'})

    if titles is None:
        titles = [f"Tensor {i+1}" for i in range(num_tensors)]

    for i, (tensor, ax) in enumerate(zip(tensors, np.atleast_1d(axes))):
        tensor = tensor.squeeze(0).detach().numpy()
        normalized_tensor = (tensor - tensor.min()) / \
                            (tensor.max() - tensor.min() + 1e-8)
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(normalized_tensor)
        colors[..., 3] = alpha

        voxels = np.ones(tensor.shape, dtype=bool)
        ax.voxels(voxels.transpose(0, 2, 1),
                  facecolors=colors.transpose(0, 2, 1, 3),
                  edgecolor='k', alpha=alpha)

        ax.set_box_aspect(aspect_ratio)
        ax.set_xlim([0, tensor.shape[0]])
        ax.set_ylim([0, tensor.shape[1]])
        ax.set_zlim([0, tensor.shape[2]])

        ax.set_xticks(np.arange(0, tensor.shape[0]+1, 1))
        ax.set_yticks(np.arange(0, tensor.shape[1]+1, 1))
        ax.set_zticks(np.arange(0, tensor.shape[2]+1, 1))

        ax.set_xlabel('Channels (C) - Feature Maps')
        ax.set_ylabel('Width (W) - Image Columns')
        ax.set_zlabel('Height (H) - Image Rows')
        ax.set_title(titles[i])

    fig.set_size_inches(fig_size[0] * num_tensors, fig_size[1])
    fig.suptitle(title, fontsize=16)
    plt.show(block=False)


# ---------- SE(2) Functions ---------- #

def se2_matrix(cos_theta, sin_theta, x, y):
    """
    Convert rotation angle θ and translation (x, y) to SE(2) matrix for batch data.

    Args:
        cos_theta (torch.Tensor): Cosine of rotation angles (B,)
        sin_theta (torch.Tensor): Sine of rotation angles (B,)
        x (torch.Tensor): Translation in x direction (B,)
        y (torch.Tensor): Translation in y direction (B,)

    Returns:
        torch.Tensor: SE(2) transformation matrices (B, 3, 3)
    """
    B = cos_theta.size(0)
    T = torch.zeros((B, 3, 3), dtype=cos_theta.dtype, device=cos_theta.device)

    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta
    T[:, 0, 2] = x
    T[:, 1, 0] = sin_theta
    T[:, 1, 1] = cos_theta
    T[:, 1, 2] = y
    T[:, 2, 2] = 1.0

    return T


def se2_geodesic_loss(T_pred, T_true):
    # Extract rotation matrices (top-left 2×2 block)
    R_pred = T_pred[:, :2, :2]
    R_true = T_true[:, :2, :2]

    # Compute rotation geodesic distance: d_R = arccos((trace(R_true^T R_pred) - 1) / 2)
    trace_term = torch.einsum('bij,bij->b', R_true, R_pred)  # batch-wise trace
    trace_term = torch.clamp((trace_term - 1) / 2, -1.0, 1.0)  # Ensure valid acos input
    d_R = torch.acos(trace_term).mean()

    # Compute translation distance: d_T = || t_true - t_pred ||
    t_pred = T_pred[:, :2, 2]  # Extract translation vector (B, 2)
    t_true = T_true[:, :2, 2]  # Extract translation vector (B, 2)
    d_T = torch.norm(t_pred - t_true, dim=1).mean()

    return d_T, d_R

import torch

def se2_chordal_loss(T_pred, T_true):

    # Compute Frobenius norm of the difference between SE(2) matrices
    frobenius_loss = torch.norm(T_pred - T_true, p='fro', dim=(1, 2)).mean()

    # Extract translation vectors (B, 2)
    t_pred = T_pred[:, :2, 2]
    t_true = T_true[:, :2, 2]

    # Compute translation L2 distance
    translation_loss = torch.norm(t_pred - t_true, dim=1).mean()

    # Combine weighted losses
    return frobenius_loss, translation_loss
