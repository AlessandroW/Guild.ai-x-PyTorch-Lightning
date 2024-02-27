import lightning as L
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import torch


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cifar10_root: str,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.cifar10_root = cifar10_root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage=None):
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_dataset = CIFAR10(root=self.cifar10_root, download=True, train=True, transform=self.transform)
        self.test_dataset = CIFAR10(root=self.cifar10_root, download=True, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self):
        raise NotImplementedError
