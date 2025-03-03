import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from examples.utils.example_utils import se2_matrix, se2_chordal_loss, log_comet_plots


class CNNLightning(pl.LightningModule):

    def __init__(self, n_classes=2, lr=1e-3, weight_decay=1e-5):
        super(CNNLightning, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # Store predictions and ground truth
        self.pred_thetas, self.pred_xs, self.pred_ys = [], [], []
        self.true_thetas, self.true_xs, self.true_ys = [], [], []

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 12, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
        )

        self.final_conv = torch.nn.Conv2d(12, 4, kernel_size=3, stride=2, padding=0)
        self.fc = torch.nn.Linear(4, 4)


        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.final_conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.fc(x)
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

        # Plot stuff for better debugging
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
