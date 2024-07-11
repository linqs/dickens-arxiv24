import torch

from torch.utils.data import Dataset


class CitationDataset(Dataset):
    def __init__(self,
                 data: list,
                 labels: list,
                 ids: list):
        self.features = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.ids[index]
