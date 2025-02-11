import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from escnn import gspaces, nn
import torchvision
import numpy as np

np.set_printoptions(precision=3, linewidth=10000, suppress=True)


### ---------- Data Handling Functions ---------- ###

def read_gabor_data(path):
    """ Reads the Gabor dataset description file. """
    return pd.read_csv(path)


def load_image_from_gabor_data(data_path, gabor_data, orientation=0.0, shift_x=0, shift_y=0):
    """ Loads an image from the Gabor dataset based on specified parameters. """
    selected_entry = gabor_data[
        (gabor_data['orientation_0'] == orientation) &
        (gabor_data['shift_x_0'] == shift_x) &
        (gabor_data['shift_y_0'] == shift_y)
    ]
    image_path = os.path.join(data_path, selected_entry['image_name'].values[0])
    return mpimg.imread(image_path), image_path


### ---------- Visualization Functions ---------- ###

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
    plt.show()


def plot_multiple_3d_tensors(tensors, titles=None, cmap="viridis", alpha=0.6, aspect_ratio=[1, 1, 1], fig_size=(8, 8)):
    """ Plots multiple 3D tensors side by side. """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(8 * num_tensors, 8), subplot_kw={'projection': '3d'})

    if titles is None:
        titles = [f"Tensor {i+1}" for i in range(num_tensors)]

    for i, (tensor, ax) in enumerate(zip(tensors, np.atleast_1d(axes))):
        tensor = tensor.squeeze(0).detach().numpy()
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(normalized_tensor)
        colors[..., 3] = alpha

        voxels = np.ones(tensor.shape, dtype=bool)
        ax.voxels(voxels.transpose(0, 2, 1), facecolors=colors.transpose(0, 2, 1, 3), edgecolor='k', alpha=alpha)

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
    plt.show()


### ---------- Equivariance Testing Functions ---------- ###

def check_equivariance_rotation(c4_conv, feat_type_in_c4, x_c4, y_c4, r2_c4_act):
    """ Tests equivariance by rotating inputs and checking output consistency. """
    for g in r2_c4_act.testing_elements:
        print(f"Testing rotation: {g}")
        x_transformed = x_c4.transform(g)
        y_from_x_transformed = c4_conv(x_transformed)
        y_transformed_from_x = y_c4.transform(g)

        plot_multiple_3d_tensors(
            [x_transformed.tensor, y_from_x_transformed.tensor, y_transformed_from_x.tensor],
            aspect_ratio=[1, 2, 2],
            titles=["x_transformed", "y_from_x_transformed", "y_transformed_from_x"],
            fig_size=(4, 4)
        )

        assert torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x.tensor, atol=1e-5), g


def check_equivariance_classic_cnn(shift_conv, feat_type_in_shift, x_shift, y_shift):
    """ Tests equivariance for a classic CNN using torchvision rotations. """
    for g in [0, 90, 180, 270]:
        print(f"Testing shift rotation: {g}")
        x_transformed = torchvision.transforms.functional.rotate(x_shift.tensor, g)
        y_from_x_transformed = shift_conv(feat_type_in_shift(x_transformed))
        y_transformed_from_x = torchvision.transforms.functional.rotate(y_shift.tensor, g)

        plot_multiple_3d_tensors(
            [x_transformed, y_from_x_transformed.tensor, y_transformed_from_x],
            aspect_ratio=[1, 2, 2],
            titles=["x_transformed", "y_from_x_transformed", "y_transformed_from_x"],
            fig_size=(4, 4)
        )

        assert torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x, atol=1e-5), g


### ---------- Main Pipeline ---------- ###

def main():
    """ Main function to execute the pipeline. """
    dataset_path = 'C:\\Users\\oat\\Datasets\\gabor_data\\C8_Z2_5'
    gabor_data = read_gabor_data(os.path.join(dataset_path, 'description.csv'))
    
    img, img_path = load_image_from_gabor_data(dataset_path, gabor_data)
    print(f"Loaded image: {img_path}")

    plot_img(img, title=r'$\mathbb{R}^2$')

    # Define symmetry groups
    r2_c4_act = gspaces.rot2dOnR2(N=4)
    r2_shift_act = gspaces.trivialOnR2()

    feat_type_in_c4 = nn.FieldType(r2_c4_act, [r2_c4_act.trivial_repr])
    feat_type_out_c4 = nn.FieldType(r2_c4_act, 3 * [r2_c4_act.regular_repr])
    c4_conv = nn.R2Conv(feat_type_in_c4, feat_type_out_c4, kernel_size=1)

    feat_type_in_shift = nn.FieldType(r2_shift_act, [r2_shift_act.trivial_repr])
    feat_type_out_shift = nn.FieldType(r2_shift_act, 3 * [r2_shift_act.regular_repr])
    shift_conv = nn.R2Conv(feat_type_in_shift, feat_type_out_shift, kernel_size=1)

    img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    x_c4 = feat_type_in_c4(img_tensor)
    x_shift = feat_type_in_shift(img_tensor)

    y_c4 = c4_conv(x_c4)
    y_shift = shift_conv(x_shift)

    plot_multiple_3d_tensors(
        [img_tensor.unsqueeze(0).squeeze(0), y_c4.tensor, y_shift.tensor],
        aspect_ratio=[1, 2, 2],
        titles=["Input", "Output tensor for C4", "Output tensor classic CNN"]
    )

    check_equivariance_rotation(c4_conv, feat_type_in_c4, x_c4, y_c4, r2_c4_act)
    check_equivariance_classic_cnn(shift_conv, feat_type_in_shift, x_shift, y_shift)


if __name__ == "__main__":
    main()
