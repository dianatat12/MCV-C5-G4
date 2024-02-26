from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F
from torchmetrics.functional import accuracy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision("high")

# data dir
train_dir = "../augmented_dataset"
test_dir = "../MIT_small_train_1/train"
validation_dir = "../test"

train_batch_size = 64
test_validation_batch_size = 128


class Model_11(pl.LightningModule):
    def __init__(self, IMG_CHANNEL, NUM_CLASSES):
        super(Model_11, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y, task="MULTICLASS", num_classes=8)
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ImageFolder(root=train_dir, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset=dataset, batch_size=train_batch_size, shuffle=True, num_workers=3
        )
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ImageFolder(root=test_dir, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=test_validation_batch_size,
            num_workers=3,
        )
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ImageFolder(root=validation_dir, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=test_validation_batch_size,
            num_workers=3,
        )
        return dataloader


if __name__ == "__main__":
    trainer = Trainer(max_epochs=100, fast_dev_run=False)
    model = Model_11(IMG_CHANNEL=3, NUM_CLASSES=8)
    trainer.fit(model)
