import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from data.dataloaders.morphomnist import io, morpho


class MnistDataset:
    def __init__(self):
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.root_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'mnist_data'
        )
        self.train_dataset = datasets.MNIST(
            self.root_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.val_dataset = datasets.MNIST(
            self.root_dir, train=False, download=True, transform=transforms.ToTensor()
        )

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )
        val_dl = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        eval_dl = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_dl, val_dl, eval_dl


class MorphoMnistDataset(MnistDataset):
    def __init__(self):
        super(MorphoMnistDataset, self).__init__()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.root_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'mnist_data',
            'plain'
        )
        self.data_path_str = "-images-idx3-ubyte.gz"
        self.label_path_str = "-labels-idx1-ubyte.gz"
        self.morpho_path_str = "-morpho.csv"

        self.train_dataset = self._create_dataset(dataset_type="train")
        self.val_dataset = self._create_dataset(dataset_type="t10k")

    def _create_dataset(self, dataset_type="train"):
        data_path = os.path.join(
            self.root_dir,
            dataset_type + self.data_path_str
        )
        label_path = os.path.join(
            self.root_dir,
            dataset_type + self.label_path_str
        )
        morpho_path = os.path.join(
            self.root_dir,
            dataset_type + self.morpho_path_str
        )
        images = io.load_idx(data_path)
        images = np.expand_dims(images, axis=1).astype('float32') / 255.0
        labels = io.load_idx(label_path)
        morpho_labels = pd.read_csv(morpho_path).values.astype('float32')
        dataset = TensorDataset(
            torch.from_numpy(images),
            torch.from_numpy(labels),
            torch.from_numpy(morpho_labels)
        )
        return dataset
