import comet_ml
import pytorch_lightning as pl
from c8_cnn import C8SteerableCNNLightning
from data.gabor_data.src.gabor_dataset_loader import GaborDataModule
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
train_val_set = "C8_Z2_128"
test_set = "C8_Z2_150"

# Initialize dataset and datamodule
data_module = GaborDataModule(
    dataset_path="C:\\Users\\oat\\Datasets\\gabor_data",
    train_val_set=train_val_set,
    test_set=test_set,
    batch_size=BATCH_SIZE,
    train_ratio=0.7,
    )

# Initialize model
model = C8SteerableCNNLightning(lr=LEARNING_RATE)

# Logger
comet_logger = CometLogger(
  api_key="TJOGq23dbkTyos8wGWpuSZ0dU",
  project_name="c8steerable-test",
  workspace="oat"
)

# Log hyperparameters
hyperparams = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "train_val_set": train_val_set,
        "test_set": test_set
}
comet_logger.log_hyperparams(hyperparams)

# Trainer setup
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
trainer = pl.Trainer(
        callbacks=[early_stop],
        max_epochs=MAX_EPOCHS,
        logger=comet_logger,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Validate the model
trainer.validate(model, datamodule=data_module)

# Test the model
trainer.test(model, datamodule=data_module)
