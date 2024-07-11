import torch


class Pathfinder(torch.nn.Module):
    """
    A torch module for pathfinding.
    The pathfinder is composed of a backbone and a multi-layer perceptron (MLP).
    """

    def __init__(self, backbone: torch.nn.Module, device="cpu"):
        super(Pathfinder, self).__init__()

        self.backbone = backbone.to(device)

        self.device = device

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)

        return x
