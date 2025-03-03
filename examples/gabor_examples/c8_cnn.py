import torch
import pytorch_lightning as pl
from escnn import gspaces
from escnn import nn
from examples.utils.example_utils import se2_matrix, se2_chordal_loss, log_comet_plots


class C8SteerableCNNLightning(pl.LightningModule):

    def __init__(self, n_classes=2, lr=1e-3, weight_decay=1e-5):
        super(C8SteerableCNNLightning, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # Store predictions and ground truth
        self.pred_thetas, self.pred_xs, self.pred_ys = [], [], []
        self.true_thetas, self.true_xs, self.true_ys = [], [], []

        # the model is equivariant under rotations by 45 degrees,
        # modeled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding
        # to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        # Define convolutional blocks
        self.block1, in_type = self._conv_block(in_type, 39, 3, 2, 1)
        self.block2, in_type = self._conv_block(in_type, 39, 3, 2, 1)
        self.block3, in_type = self._conv_block(in_type, 39, 3, 2, 1)
        self.block4, in_type = self._conv_block(in_type, 39, 3, 1, 1)
        self.block5, in_type = self._conv_block(in_type, 39, 3, 1, 1)
        self.block6, in_type = self._conv_block(in_type, 39, 3, 1, 1)

        self.final_conv, in_type = self._conv_block(in_type, 64, 3, 2, 0)

        self.fc1 = torch.nn.Linear(512, 39)
        self.fc2 = torch.nn.Linear(39, 32)
        self.fc3 = torch.nn.Linear(32, 4)

        self.save_hyperparameters()

    def _conv_block(self, in_type, out_channels, kernel_size, stride, padding):
        out_type = nn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
        block = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        return block, out_type

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.final_conv(x)
        x = x.tensor.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # Normalize first two outputs (cos thetha, sin theta)
        x = torch.cat([torch.nn.functional.normalize(x[:, :2], dim=1), x[:, 2:]], dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        T_pred = se2_matrix(logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3])
        T_target = se2_matrix(y[:, 0], y[:, 1], y[:, 2], y[:, 3])

        rot_loss, shift_loss = se2_chordal_loss(T_pred, T_target)

        self.log('train_rot_loss', rot_loss)
        self.log('train_shift_loss', shift_loss)

        loss = shift_loss + 10*rot_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        T_pred = se2_matrix(logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3])
        T_target = se2_matrix(y[:, 0], y[:, 1], y[:, 2], y[:, 3])

        rot_loss, shift_loss = se2_chordal_loss(T_pred, T_target)

        self.log('val_rot_loss', rot_loss)
        self.log('val_shift_loss', shift_loss)

        loss = shift_loss + 10*rot_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        T_pred = se2_matrix(logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3])
        T_target = se2_matrix(y[:, 0], y[:, 1], y[:, 2], y[:, 3])

        rot_loss, shift_loss = se2_chordal_loss(T_pred, T_target)

        self.log('test_rot_loss', rot_loss)
        self.log('test_shift_loss', shift_loss)

        loss = shift_loss + 100*rot_loss
        self.log('test_loss', loss)

        pred_theta = torch.atan2(logits[:, 1], logits[:, 0]) * (180 / torch.pi)
        true_theta = torch.atan2(y[:, 1], y[:, 0]) * (180 / torch.pi)

        pred_x, pred_y = logits[:, 2], logits[:, 3]
        true_x, true_y = y[:, 2], y[:, 3]

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
