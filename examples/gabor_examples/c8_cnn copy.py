import torch
import pytorch_lightning as pl
from escnn import gspaces
from escnn.group import so2_group
from escnn import nn
import torch.nn.functional as F
from examples.utils.example_utils import se2_matrix, se2_chordal_loss, log_comet_plots, se2_geodesic_loss


class C8SteerableCNNLightning(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5,
                 group_order=8, input_channels=1, hidden_channels=24):
        super(C8SteerableCNNLightning, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay


        # Store predictions and ground truth
        self.pred_thetas, self.pred_xs, self.pred_ys = [], [], []
        self.true_thetas, self.true_xs, self.true_ys = [], [], []

        # Define SE(2) Rotation Group (Cyclic Group of Order 8)
        self.r2_act = gspaces.rot2dOnR2(N=group_order)

        # Input Type: Trivial representation (standard image input)
        self.input_type = nn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

        # Hidden Type: Regular representation (transforms under SE(2))
        self.hidden_type = nn.FieldType(self.r2_act, hidden_channels * [self.r2_act.regular_repr])

        # Output Type: Irreducible representation for se(2)
        self.out_shift = nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr])  # (x, y) equivariant vector
        self.out_theta = nn.FieldType(self.r2_act, 2*[self.r2_act.irrep(1)])  # θ (scalar)

        # Define convolutional blocks (With Stride for Spatial Reduction)
        self.block1, in_type = self._conv_block(self.input_type, self.hidden_type, kernel_size=5, stride=3, padding=2)
        self.block2, in_type = self._conv_block(self.hidden_type, self.hidden_type, kernel_size=3, stride=2, padding=2)
        self.block3, in_type = self._conv_block(self.hidden_type, self.hidden_type, kernel_size=3, stride=2, padding=2)
        self.block4, in_type = self._conv_block(self.hidden_type, self.hidden_type, kernel_size=3, stride=2, padding=2)
        self.block5, in_type = self._conv_block(self.hidden_type, self.hidden_type, kernel_size=3, stride=3, padding=2)

        # **Final Equivariant Projection using Convolutions Instead of Pooling**
        self.reduction_conv = nn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, stride=2, padding=1)

        # **Final SE(2) → se(2) Mapping**
        self.conv_shift = nn.R2Conv(self.hidden_type, self.out_shift, kernel_size=1)
        self.conv_theta = nn.R2Conv(self.hidden_type, self.out_theta, kernel_size=1)

        self.save_hyperparameters()

    def _conv_block(self, in_type, out_type, kernel_size, stride, padding=1):
        # out_type = nn.FieldType(self.r2_act, out_channels )
        block = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, stride=stride,bias=True),
            # nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )
        return block, out_type

    def forward(self, x):
        x = nn.GeometricTensor(x, self.input_type)  # Convert to ESCNN GeometricTensor

        # Apply Equivariant Convolutions
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # Spatially reduced
        x = self.block4(x)  # Further reduction
        x = self.block5(x)  # Further reduction

        # **Use Convolution for Final Spatial Reduction Instead of Softmax**
        x = self.reduction_conv(x)  # Reduces spatial dimensions further

        # Convert back to GeometricTensor for final equivariant projection
        # x = nn.GeometricTensor(x.tensor.mean(dim=[2, 3], keepdim=True), self.hidden_type)

        # **Final Equivariant Projection (SE(2) → se(2))**
        shift_pred = self.conv_shift(x)  # Predict (x, y) position
        theta_pred = self.conv_theta(x)  # Predict θ rotation

        # Convert GeometricTensors to standard tensor output
        return torch.cat([theta_pred.tensor, shift_pred.tensor], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        rot_loss, shift_loss, loss = se2_geodesic_loss(logits, y)

        self.log('train_rot_loss', rot_loss)
        self.log('train_shift_loss', shift_loss)
        self.log('train_loss', loss)

        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item()}")  # Check gradient magnitudes

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        rot_loss, shift_loss, loss = se2_geodesic_loss(logits, y)

        self.log('val_rot_loss', rot_loss)
        self.log('val_shift_loss', shift_loss)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        rot_loss, shift_loss, loss = se2_geodesic_loss(logits, y)

        self.log('test_rot_loss', rot_loss)
        self.log('test_shift_loss', shift_loss)
        self.log('test_loss', loss)

        # Plot stuff for better debugging
        # pred_theta = torch.atan2(logits[:, 1], logits[:, 0]) * (180 / torch.pi)
        # true_theta = torch.atan2(y[:, 1], y[:, 0]) * (180 / torch.pi)
        pred_theta = logits[:, 0]
        true_theta = y[:, 0]

        pred_x, pred_y = logits[:, 1], logits[:, 2]
        true_x, true_y = y[:, 1], y[:, 2]

        self.pred_thetas.extend(pred_theta.cpu().numpy())
        self.pred_xs.extend(pred_x.cpu().numpy())
        self.pred_ys.extend(pred_y.cpu().numpy())

        self.true_thetas.extend(true_theta.cpu().numpy())
        self.true_xs.extend(true_x.cpu().numpy())
        self.true_ys.extend(true_y.cpu().numpy())

        # Log metrics in Comet
        self.logger.experiment.log_metric("test_theta_error", torch.abs(pred_theta - true_theta).mean().item())
        self.logger.experiment.log_metric("test_x_error", torch.abs(pred_x - true_x).mean().item())
        self.logger.experiment.log_metric("test_y_error", torch.abs(pred_y - true_y).mean().item())

        return loss

    def on_test_epoch_end(self):
        """Called at the end of the test epoch to log final predictions in Comet."""
        log_comet_plots(
            self.logger,
            true_thetas=self.true_thetas,
            pred_thetas=self.pred_thetas,
            true_xs=self.true_xs,
            pred_xs=self.pred_xs,
            true_ys=self.true_ys,
            pred_ys=self.pred_ys
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay
                                )
