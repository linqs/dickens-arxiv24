import numpy as np
import os
import torch


def save_model(model, data_dir: str, model_name: str):
    os.makedirs(f"{data_dir}/saved-networks", exist_ok=True)
    torch.save(model.state_dict(), f"{data_dir}/saved-networks/{model_name}")


def load_model(model, data_dir: str, model_name: str):
    model.load_state_dict(torch.load(f"{data_dir}/saved-networks/{model_name}", map_location=model.device))
    return model


def sample_gumbel(shape, device, eps=1e-12):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=1)


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature, hard=False):
    y = gumbel_sigmoid_sample(logits, temperature)

    if not hard:
        return y

    return (y > 0.5).float()


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumble-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def get_temperature(initial_temperature, min_temperature, epoch, rate=1.0e-5):
    return max(min_temperature, initial_temperature * np.exp(-1.0 * rate * epoch))


def get_torch_device():
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # else:
    #     return torch.device("cpu")
    return torch.device("cpu")
