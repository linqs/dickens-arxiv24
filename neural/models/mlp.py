import torch


class MLP(torch.nn.Module):
    """
    A multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Dropout(dropout_rate))
        for i in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(input_dim if i == 0 else h[i - 1], h[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout_rate))
        self.layers.append(torch.nn.Linear(input_dim if num_layers == 1 else h[-1], output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
