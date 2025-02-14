import torch
import pytorch_lightning as pl
from escnn import gspaces
from escnn import nn
from examples.utils.example_utils import se2_matrix, se2_chordal_loss


class C8SteerableCNNLightning(pl.LightningModule):

    def __init__(self, n_classes=2, lr=1e-3, weight_decay=1e-5):
        super(C8SteerableCNNLightning, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # the model is equivariant under rotations by 45 degrees,
        # modeled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding
        # to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        # Define convolutional blocks
        self.block1, in_type = self._conv_block(in_type, 12, 7, 2)
        self.block2, in_type = self._conv_block(in_type, 24, 5, 2)
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(in_type, sigma=0.66, stride=2)
        )

        # self.block3, in_type = self._conv_block(in_type, 24, 5, 2)
        # self.block4, in_type = self._conv_block(in_type, 48, 5, 2)
        # self.pool2 = nn.SequentialModule(
        #     nn.PointwiseAvgPoolAntialiased(in_type, sigma=0.66, stride=2)
        # )

        self.block5, in_type = self._conv_block(in_type, 48, 5, 2)
        self.block6, in_type = self._conv_block(in_type, 32, 5, 1)

        self.pool3 = nn.PointwiseAvgPoolAntialiased(in_type, sigma=0.66, stride=2, padding=0)

        self.final_conv = nn.R2Conv(in_type, nn.FieldType(self.r2_act, 4 * [self.r2_act.trivial_repr]), kernel_size=3, stride=2)
        self.dropout = nn.PointwiseDropout(in_type, p=0.5)

        self.save_hyperparameters()

    def _conv_block(self, in_type, out_channels, kernel_size, padding):
        out_type = nn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
        block = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        return block, out_type

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.final_conv(x)
        x = x.tensor.squeeze(2).squeeze(2)
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

        loss = rot_loss*10 + shift_loss
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

        loss = rot_loss*10 + shift_loss
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

        loss = rot_loss*10 + shift_loss
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay
                                )
