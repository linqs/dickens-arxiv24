import os
import sys
import typing

import torch
import torchvision
import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import v2

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

from neural.models.mlp import MLP

AUGMENTATIONS = {
    "ElasticTransform": v2.ElasticTransform,
    "RandomAffine": v2.RandomAffine,
    "RandomResizedCrop": v2.RandomResizedCrop,
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "ColorJitter": v2.ColorJitter,
    "RandomGrayscale": torchvision.transforms.RandomGrayscale
}


class SimCLR(object):
    def __init__(self, model, backbone, projection_head, optimizer, scheduler, temperature, num_augmentations, device):
        self.model = model
        self.backbone = backbone
        self.projection_head = projection_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.device = device
        self.num_augmentations = num_augmentations
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def cosine_similarity(self, embedding):
        normalized_embedding = torch.nn.functional.normalize(embedding, dim=1)
        return torch.matmul(normalized_embedding, normalized_embedding.T) / self.temperature

    def create_same_labels_mask(self, labels):
        return torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()

    def train(self, train_loader, epochs):
        """
        Train the model.
        :param train_loader: The training data loader. Expected to return a tuple of (input, labels).
            Labels are used to create positive and negative pairs and should not be one-hot encoded.
        :param epochs: The number of epochs to train for.
        """
        for epoch_counter in range(epochs):
            epoch_loss = 0
            with tqdm.tqdm(train_loader) as tq:
                tq.set_description("Epoch:{}".format(epoch_counter))
                for step, batch in enumerate(tq):
                    input, labels = batch

                    input = torch.cat(input, dim=0)
                    labels = torch.cat(labels, dim=0)

                    # Move to device.
                    input = input.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass.
                    embedding = self.model(input)

                    # Compute loss.
                    cosine_similarity_matrix = self.cosine_similarity(embedding)
                    same_image_mask = self.create_same_labels_mask(labels)
                    loss = self.criterion(cosine_similarity_matrix, same_image_mask)
                    epoch_loss += loss.item()

                    # Backward pass.
                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                    tq.set_postfix(loss=epoch_loss / ((step + 1) * train_loader.batch_size))

            self.scheduler.step()


def pretrain_simclr(train_dataset: Dataset, experiment_dir: str, config: dict,
                    backbone: torch.nn.Module, model: typing.Callable,
                    hidden_dim=128, epochs: int = 500, batch_size: int = 1024,
                    learning_rate: float = 3.0e-3, weight_decay: float = 1.0e-6,
                    temperature: float = 0.5, num_augmentations: int = 2,
                    device: torch.device = torch.device("cpu")):
    """
    Pretrain a model using SimCLR.
    :param train_dataset: The dataset to pretrain on.
    :param experiment_dir: The directory to save the model to.
    :param config: The configuration for the augmentations.
    :param backbone: The backbone to use for the model.
    :param model: The model to connect the backbone to.
        Expected to be a callable that takes the backbone and a projection head as arguments.
    :param hidden_dim: The hidden dimension of the projection head.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :param learning_rate: The learning rate to use.
    :param weight_decay: The weight decay to use.
    :param temperature: The temperature to use for the contrastive loss.
    :param num_augmentations: The number of augmentations to use.
    :param device: The device to use.
    :return: The SimCLR pretrained backbone and the projection head.
    """
    print(f"Pretraining on {experiment_dir}...")

    contrastive_learning_dataset = ContrastiveLearningDataset(train_dataset, config)
    train_loader = torch.utils.data.DataLoader(
        contrastive_learning_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    backbone = backbone.to(device)
    pretrained_projection_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2).to(device)
    model = model(backbone, pretrained_projection_head, device=device)

    # Initialize learning components.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    simclr = SimCLR(model, backbone, pretrained_projection_head, optimizer, scheduler,
                    temperature, num_augmentations, device)

    # Train the model.
    simclr.train(train_loader, epochs)

    return backbone, pretrained_projection_head


class ContrastiveLearningDataset(Dataset):
    def __init__(self, dataset, config):
        self.feature_transform = ContrastiveLearningViewGenerator(torchvision.transforms.Compose(
            [AUGMENTATIONS[config["augmentations"][i][0]](**config["augmentations"][i][1]) for i in range(len(config["augmentations"]))]
        ))
        self.label_transform = ContrastiveLearningViewGenerator(torchvision.transforms.Compose([]), augmentations=2)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.feature_transform(self.dataset[index][0]), self.label_transform(self.dataset[index][1])

    def __len__(self):
        return len(self.dataset)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, augmentations=2):
        self.base_transform = base_transform
        self.augmentations = augmentations

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.augmentations)]
