import comet_ml
import pytorch_lightning as pl
from c8_cnn import C8SteerableCNNLightning
from data.gabor_data.src.gabor_dataset_loader import GaborDataModule
from pytorch_lightning.loggers import CometLogger

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 400

# Initialize dataset and datamodule
data_module = GaborDataModule(
    dataset_path="C:\\Users\\oat\\Datasets\\gabor_data",
    train_val_set="C4_Z2_100",
    test_set="C8_Z2_150",
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
        "max_epochs": MAX_EPOCHS
}
comet_logger.log_hyperparams(hyperparams)

# Trainer setup
trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=comet_logger,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Validate the model
trainer.validate(model, datamodule=data_module)

# Test the model
trainer.test(model, datamodule=data_module)
