import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchvision

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

from neural.models.mlp import MLP

DATA_DIR = f"{THIS_DIR}/../data"


def main(fold):
    # Load the simclr pretrained model.
    backbone = torchvision.models.resnet18(num_classes=128)

    data_folder = f"{DATA_DIR}/mnist-1/split::0/train-size::00600/unlabeled::1.00/overlap::0.00"

    backbone.load_state_dict(torch.load(
        f"{data_folder}/saved-networks/simclr-pretrained-backbone.pt"
    ))

    projection_head = MLP(128, 128, 128, 2)
    projection_head.load_state_dict(torch.load(
        f"{data_folder}/saved-networks/simclr-pretrained-projection-head.pt"
    ))

    # Load the mnist dataset.
    indices = torch.tensor(
        pd.read_csv(f"{data_folder}/image-target-{fold}.txt", sep="\t", header=None)[0].unique())
    all_features = pd.read_csv(f"{data_folder}/neural-data-{fold}.txt", sep="\t", header=None, index_col=0)
    features = torch.tensor(all_features.loc[indices].iloc[:, :-2].values, dtype=torch.float32)
    labels = all_features.loc[indices].iloc[:, -2].values
    features = features.reshape(-1, 1, 28, 28)
    features = features.expand(-1, 3, -1, -1)

    # Get the features from the backbone.
    features = backbone(features)

    # Dimensionality reduction.
    dim_reduced_features = PCA(n_components=10).fit_transform(features.detach().numpy())

    # Cluster the features.
    kmeans_clustering = KMeans(n_clusters=10, n_init='auto').fit_predict(dim_reduced_features)

    # Save the clusters.
    image_id_to_cluster = pd.DataFrame(np.stack([indices, kmeans_clustering], axis=1), columns=["image_id", "cluster"])
    image_id_to_cluster.to_csv(f"{data_folder}/simclr-clusters-{fold}.txt", sep="\t", index=False, header=False)

    # Plot the clusters.
    two_d_features = TSNE(n_components=2).fit_transform(dim_reduced_features)

    fig, ax = plt.subplots()
    for i in range(10):
        ax.scatter(two_d_features[labels == i, 0], two_d_features[labels == i, 1], label=i)
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    for fold in ["train", "valid", "test"]:
        main(fold)
