import torch
import os
from escnn import gspaces, nn
import torchvision
import numpy as np
from examples.utils.example_utils import read_gabor_data, \
    load_image_from_gabor_data, plot_img, plot_multiple_3d_tensors

np.set_printoptions(precision=3, linewidth=10000, suppress=True)


### ---------- Equivariance Testing Functions ---------- ###

def c4_check_equivariance_rotation(c4_conv, feat_type_in_c4, x_c4, y_c4, r2_c4_act):
    """ Tests equivariance by rotating inputs and checking output consistency. """
    for g in r2_c4_act.testing_elements:
        print(f"Testing rotation: {g}")
        x_transformed = x_c4.transform(g)
        y_from_x_transformed = c4_conv(x_transformed)
        y_transformed_from_x = y_c4.transform(g)

        plot_multiple_3d_tensors(
            [x_transformed.tensor, y_from_x_transformed.tensor, y_transformed_from_x.tensor],
            aspect_ratio=[1, 2, 2],
            title=f"C4 Equivariance Test, rotation: {g}",
            titles=["x_transformed", "y_from_x_transformed", "y_transformed_from_x"],
            fig_size=(4, 4)
        )

        try:
            assert torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x.tensor, atol=1e-5), g
        except AssertionError as e:
            print(f"C4 Equivariance test failed for rotation {g}: {e}")


def cnn_check_equivariance_rotation(shift_conv, feat_type_in_shift, x_shift, y_shift):
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
            title=f"Classic CNN Equivariance Test, rotation: {g}",
            fig_size=(4, 4)
        )

        try:
            assert torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x, atol=1e-5), g
        except AssertionError as e:
            print(f"CNN Equivariance test failed for rotation {g}: {e}")


### ---------- Main Pipeline ---------- ###

def main():
    """ Main function to execute the pipeline. """
    dataset_path = 'C:\\Users\\oat\\Datasets\\gabor_data\\C4'
    gabor_data = read_gabor_data(os.path.join(dataset_path, 'description.csv'))

    img, img_path = load_image_from_gabor_data(dataset_path, gabor_data)
    print(f"Loaded image: {img_path}")

    plot_img(img, title=r'$\mathbb{R}^2$')

    # Define symmetry groups
    r2_c4_act = gspaces.rot2dOnR2(N=4)  # C4 group
    r2_shift_act = gspaces.trivialOnR2()  # Classic CNN

    # Define C4 features and layers
    feat_type_in_c4 = nn.FieldType(r2_c4_act, [r2_c4_act.trivial_repr])
    feat_type_out_c4 = nn.FieldType(r2_c4_act, 1 * [r2_c4_act.regular_repr])
    c4_conv = nn.R2Conv(feat_type_in_c4, feat_type_out_c4, kernel_size=3)

    # Define classic CNN features and layers
    feat_type_in_shift = nn.FieldType(r2_shift_act, [r2_shift_act.trivial_repr])
    feat_type_out_shift = nn.FieldType(r2_shift_act, 1 * [r2_shift_act.regular_repr])
    shift_conv = nn.R2Conv(feat_type_in_shift, feat_type_out_shift, kernel_size=3)
    # torch_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
    sobel_kernel = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]], dtype=torch.float32)
    sobel_kernel = sobel_kernel.view(1, 1, 3, 3)
    shift_conv.filter.data = sobel_kernel

    # Create input tensors
    img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    x_c4 = feat_type_in_c4(img_tensor)
    x_shift = feat_type_in_shift(img_tensor)

    # Forward pass
    y_c4 = c4_conv(x_c4)
    y_shift = shift_conv(x_shift)

    y_c4_max = y_c4.tensor.max()
    y_shift_max = y_shift.tensor.max()
    print(f"Max value in C4 output: {y_c4_max}")
    print(f"Max value in CNN output: {y_shift_max}")

    y_c4_min = y_c4.tensor.min()
    y_shift_min = y_shift.tensor.min()
    print(f"Min value in C4 output: {y_c4_min}")
    print(f"Min value in CNN output: {y_shift_min}")

    # Visualize results
    plot_multiple_3d_tensors(
        [img_tensor.unsqueeze(0).squeeze(0), y_c4.tensor, y_shift.tensor],
        aspect_ratio=[1, 2, 2],
        titles=["Input", "Output tensor for C4", "Output tensor classic CNN"],
        title="Output tensors",
    )

    c4_check_equivariance_rotation(c4_conv, feat_type_in_c4, x_c4, y_c4, r2_c4_act)
    cnn_check_equivariance_rotation(shift_conv, feat_type_in_shift, x_shift, y_shift)

    # hold plots
    input("Press Enter to continue...")



if __name__ == "__main__":
    main()
