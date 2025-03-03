import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch.nn.functional as F
import matplotlib.colors as mcolors
import hashlib


# ---------- Data Handling Functions ---------- #

# def se2_geodesic_loss( pred, target):
#     """
#     Compute geodesic loss between predicted and ground-truth SE(2) poses.

#     Args:
#     - pred: Tensor of shape (batch_size, 3) with predicted (x, y, theta)
#     - gt: Tensor of shape (batch_size, 3) with ground truth (x, y, theta)

#     Returns:
#     - loss (Tensor): SE(2) geodesic loss.
#     """
#     pred_theta, pred_xy = pred[:, :2], pred[:, 2:]
#     pred_theta = torch.atan2(pred_theta[:, 1], pred_theta[:, 0])
#     target_theta, target_xy = target[:, 0], target[:, 1:3]

#     # Translation error (Euclidean distance)
#     trans_loss = torch.norm(pred_xy - target_xy, dim=1).mean()

#     # Rotation error (Small-angle geodesic distance)
#     # rot_loss = torch.abs(torch.atan2(torch.sin(pred_theta - target_theta),
#     #                                 torch.cos(pred_theta - target_theta))).mean()
#     rot_loss = torch.abs(torch.atan2(torch.sin(pred_theta - target_theta),
#                          torch.cos(pred_theta - target_theta))).mean()

#     # Total geodesic loss
#     loss = trans_loss + 10*rot_loss

#     return rot_loss, trans_loss, loss
    

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


def generate_fixed_colormap(cmap_name, num_colors):
    """
    Generates a fixed discrete colormap with a predefined number of colors.

    :param cmap_name: Name of the colormap to use
    :param num_colors: Number of discrete colors
    :return: ListedColormap with fixed colors
    """
    np.random.seed(42)  # Fix seed for consistent colors
    cmap = plt.get_cmap(cmap_name, num_colors)
    return cmap


def value_to_fixed_color(value, colormap, num_colors):
    """
    Assigns a fixed color to a given value using a hash function.

    :param value: Numerical value to be mapped to a color
    :param colormap: Pre-generated colormap with fixed colors
    :param num_colors: Number of discrete colors
    :return: RGBA color
    """
    hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    color_index = hash_value % num_colors  # Map hash to a color index
    return colormap(color_index)


def plot_multiple_3d_tensors(tensors,
                             titles=None, cmap="gist_ncar",
                             title=None, alpha=0.9,
                             aspect_ratio=[1, 1, 1],
                             fig_size=(8, 8),
                             num_colors=256):
    """
    Plots multiple 3D tensors side by side with a fixed color mapping.
    Ensures the same values always map to the same colors across different runs.

    :param tensors: List of 3D tensors (each of shape (1, C, H, W))
    :param titles: List of titles corresponding to each tensor
    :param cmap: Colormap to use for coloring (default: 'gist_ncar')
    :param title: Overall figure title
    :param alpha: Transparency of the voxels
    :param aspect_ratio: Aspect ratio for the voxel plots
    :param fig_size: Figure size
    :param num_colors: Number of discrete colors for mapping
    """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors,
                             figsize=(8 * num_tensors, 8),
                             subplot_kw={'projection': '3d'})

    if titles is None:
        titles = [f"Tensor {i+1}" for i in range(num_tensors)]

    # Generate a fixed colormap (ensures deterministic color mapping)
    fixed_colormap = generate_fixed_colormap(cmap, num_colors)

    for i, (tensor, ax) in enumerate(zip(tensors, np.atleast_1d(axes))):
        tensor = tensor.squeeze(0).detach().numpy()

        # Map tensor values to fixed colors using hashing
        colors = np.zeros((*tensor.shape, 4))  # Store RGBA colors
        unique_values = np.unique(tensor)  # Get unique values in the tensor

        # Assign deterministic colors to each unique value
        color_map = {val: value_to_fixed_color(val, fixed_colormap, num_colors) for val in unique_values}

        # Apply the fixed colors to the tensor
        for val in unique_values:
            colors[tensor == val] = color_map[val]  # Assign the same color to the same value

        colors[..., 3] = alpha  # Set transparency

        # Create voxel grid
        voxels = np.ones(tensor.shape, dtype=bool)

        # Plot voxels with assigned colors and transparency
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

    # Compute rotation geodesic distance
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_true)
    # Convert R_diff to axis-angle representation
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1]
    theta = torch.acos((trace - 1) / 2)
    d_R = theta.mean()

    # Compute translation distance: d_T = || t_true - t_pred ||
    t_pred = T_pred[:, :2, 2]  # Extract translation vector (B, 2)
    t_true = T_true[:, :2, 2]  # Extract translation vector (B, 2)
    d_T = torch.norm(t_pred - t_true, dim=1).mean()

    return d_T, d_R


def se2_chordal_loss(T_pred, T_true):

    # Compute Frobenius norm of the difference between SO(2) matrices
    T_true_rot = T_true[:, :2, :2]
    frobenius_loss = torch.norm(T_pred[:,:2,:2] - T_true[:,:2,:2], p='fro', dim=(1, 2)).mean()

    # Extract translation vectors (B, 2)
    t_pred = T_pred[:, :2, 2]
    t_true = T_true[:, :2, 2]

    # Compute translation L2 distance
    translation_loss = torch.norm(t_pred - t_true, dim=1).mean()

    # Combine weighted losses
    return frobenius_loss, translation_loss

def plot_predictions(true_thetas, pred_thetas, true_xs, pred_xs, true_ys, pred_ys):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot True vs. Predicted Theta
    axes[0].scatter(true_thetas, pred_thetas, alpha=0.7, label="Predictions")
    axes[0].plot([-180, 180], [-180, 180], 'r--', label="Ideal")  # Ideal line
    axes[0].set_xlabel("True Theta (degrees)")
    axes[0].set_ylabel("Predicted Theta (degrees)")
    axes[0].set_title("Theta Prediction vs. Ground Truth")
    axes[0].grid()
    axes[0].legend()

    # Plot True vs. Predicted X
    axes[1].scatter(true_xs, pred_xs, alpha=0.7, label="Predictions")
    axes[1].plot([-1, 1], [-1, 1], 'r--', label="Ideal")  # Ideal line
    axes[1].set_xlabel("True X")
    axes[1].set_ylabel("Predicted X")
    axes[1].set_title("X Prediction vs. Ground Truth")
    axes[1].grid()
    axes[1].legend()

    # Plot True vs. Predicted Y
    axes[2].scatter(true_ys, pred_ys, alpha=0.7, label="Predictions")
    axes[2].plot([-1, 1], [-1, 1], 'r--', label="Ideal")  # Ideal line
    axes[2].set_xlabel("True Y")
    axes[2].set_ylabel("Predicted Y")
    axes[2].set_title("Y Prediction vs. Ground Truth")
    axes[2].grid()
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def log_comet_plots(logger, true_thetas, pred_thetas, true_xs, pred_xs, true_ys, pred_ys):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot True vs. Predicted Theta
    axes[0].scatter(true_thetas, pred_thetas, alpha=0.7, label="Predictions")
    true_thetas = np.array(true_thetas)
    axes[0].plot(
        [true_thetas.min(), true_thetas.max()],
        [true_thetas.min(), true_thetas.max()],
        'r--', label="Ideal"
    )  # Ideal line
    axes[0].set_xlabel("True Theta (degrees)")
    axes[0].set_ylabel("Predicted Theta (degrees)")
    axes[0].set_title("Theta Prediction vs. Ground Truth")
    axes[0].grid()
    axes[0].legend()

    # Plot True vs. Predicted X
    axes[1].scatter(true_xs, pred_xs, alpha=0.7, label="Predictions")
    true_xs = np.array(true_xs)
    axes[1].plot([true_xs.min(), true_xs.max()], [true_xs.min(), true_xs.max()], 'r--', label="Ideal")  # Ideal line
    axes[1].set_xlabel("True X")
    axes[1].set_ylabel("Predicted X")
    axes[1].set_title("X Prediction vs. Ground Truth")
    axes[1].grid()
    axes[1].legend()

    # Plot True vs. Predicted Y
    axes[2].scatter(true_ys, pred_ys, alpha=0.7, label="Predictions")
    true_ys = np.array(true_ys)
    axes[2].plot([true_ys.min(), true_ys.max()], [true_ys.min(), true_ys.max()], 'r--', label="Ideal")  # Ideal line
    axes[2].set_xlabel("True Y")
    axes[2].set_ylabel("Predicted Y")
    axes[2].set_title("Y Prediction vs. Ground Truth")
    axes[2].grid()
    axes[2].legend()

    plt.tight_layout()

    # Save plot and log to Comet
    # plt.savefig("test_predictions.png")
    # logger.experiment.log_image("test_predictions", "test_predictions.png")
    logger.experiment.log_figure(figure_name="test_predictions", figure=fig)
    logger.experiment.end()



