import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class CNNLightning(pl.LightningModule):

    def __init__(self, n_classes=2, lr=1e-3):
        super(CNNLightning, self).__init__()
        self.lr = lr

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 24, kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 48, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(48, 96, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 64, kernel_size=5, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=1, stride=1)
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )


        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.final_conv(x)
        return x.squeeze(2).squeeze(2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        rot_loss = torch.nn.functional.mse_loss(logits[0], y[0])
        shift_loss = torch.nn.functional.mse_loss(logits[1:], y[1:])
        self.log('train_rot_loss', rot_loss)
        self.log('train_shift_loss', shift_loss)
        return rot_loss + shift_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        rot_loss = torch.nn.functional.mse_loss(logits[0], y[0])
        shift_loss = torch.nn.functional.mse_loss(logits[1:], y[1:])
        self.log('val_rot_loss', rot_loss)
        self.log('val_shift_loss', shift_loss)
        return rot_loss + shift_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        rot_loss = torch.nn.functional.mse_loss(logits[0], y[0])
        shift_loss = torch.nn.functional.mse_loss(logits[1:], y[1:])
        self.log('test_rot_loss', rot_loss)
        self.log('test_shift_loss', shift_loss)
        return rot_loss + shift_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
