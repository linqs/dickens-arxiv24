import torch


class MNISTClassifier(torch.nn.Module):
    """
    A torch module for MNIST classification.
    The MNIST classifier is composed of a backbone and a multi-layer perceptron (MLP).
    """

    def __init__(self, backbone: torch.nn.Module, mlp: torch.nn.Module, device="cpu"):
        super(MNISTClassifier, self).__init__()

        self.backbone = backbone.to(device)
        self.mlp = mlp.to(device)

        self.device = device

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        x = self.mlp(x)

        return x
