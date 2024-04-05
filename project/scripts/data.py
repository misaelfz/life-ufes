import os
import pickle
import shutil
from random import choice
import matplotlib.pyplot as plt
from scipy import stats, spatial
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import no_grad, tensor, softmax
from torchvision.transforms import ToTensor, v2
from sklearn.model_selection import train_test_split


def get_dataset(path, target):
    file = os.path.join(path, f"{target}.pkl")
    assert os.path.exists(file)

    with open(file, "rb") as handle:
        return pickle.load(handle)


def save_mnist(path, batch_size):
    file = os.path.join(path, "mnist.pkl")

    if not os.path.exists(file):
        train = MNIST(root="download", train=True, download=True, transform=ToTensor())
        test = MNIST(root="download", train=False, download=True, transform=ToTensor())
        shutil.rmtree("download")

        train, val = train_test_split(
            train, train_size=50000, test_size=10000, random_state=13
        )

        mnist = {
            "train": DataLoader(train, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test, batch_size=batch_size, shuffle=False),
        }

        with open(file, "wb") as handle:
            pickle.dump(mnist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_augmented(path, batch_size):
    file = os.path.join(path, "augmented.pkl")

    if not os.path.exists(file):
        mnist = get_dataset(path, "mnist")
        recrop = v2.RandomResizedCrop(28, (0.5, 0.5))
        rotate = lambda x: v2.RandomRotation(degrees=choice([(-45, -45), (45, 45)]))(x)

        augmented = []

        for image, label in mnist["train"].dataset:
            augmented.append((image, label))

            for _ in range(2):
                augmented.append((recrop(image), label))
                augmented.append((rotate(image), label))

        augmented = DataLoader(augmented, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_classified(path, model):
    file = os.path.join(path, "classified.pkl")

    if not os.path.exists(file):
        augmented = get_dataset(path, "augmented")
        classified = {}

        model.eval()
        with no_grad():
            for images, labels in augmented:
                logits, embeddings = model(images)

                for i in range(len(images)):
                    image = images[i].numpy()
                    label = labels[i].item()
                    logit = logits[i].numpy()
                    embedding = embeddings[i].numpy()

                    if label not in classified:
                        classified[label] = [(image, label, logit, embedding)]
                    else:
                        classified[label].append((image, label, logit, embedding))

            with open(file, "wb") as handle:
                pickle.dump(classified, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_random(path, batch_size, subset_ratio):
    file = os.path.join(path, "random.pkl")

    if not os.path.exists(file):
        classified = get_dataset(path, "classified")

        random = []
        size = int(
            (sum(len(v) for v in classified.values()) * subset_ratio) / len(classified)
        )

        for _ in range(size):
            for key in classified.keys():
                index = choice(range(len(classified[key])))
                image, label, _, _ = classified[key].pop(index)
                random.append((image, label))

        random = DataLoader(random, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(random, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_entropy(path, batch_size, subset_ratio):
    file = os.path.join(path, "entropy.pkl")

    if not os.path.exists(file):
        classified = get_dataset(path, "classified")

        for key in classified:
            for i in range(len(classified[key])):
                image, label, logits, embeddings = classified[key][i]
                probabilities = softmax(tensor(logits), dim=0).numpy()
                entropy = stats.entropy(pk=probabilities, base=2)
                classified[key][i] = (entropy, (image, label, logits, embeddings))

            classified[key].sort(key=lambda x: x[0], reverse=True)

        entropy = []
        size = int(
            (sum(len(v) for v in classified.values()) * subset_ratio) / len(classified)
        )

        for _ in range(size):
            for key in classified.keys():
                _, item = classified[key].pop(0)
                image, label, _, _ = item
                entropy.append((image, label))

        entropy = DataLoader(entropy, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(entropy, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_datasets(path):
    files = [
        "mnist.pkl",
        "augmented.pkl",
        "classified.pkl",
        "random.pkl",
        "entropy.pkl",
    ]

    for file in files:
        assert os.path.exists(os.path.join(path, file))

    print("All datasets are available.")


def plot_batch(dataset):
    for images, labels in dataset:
        rows = 10
        cols = 10

        plt.figure(figsize=(40, 4 * rows))

        for i in range(rows * cols):
            plt.subplot(rows, cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i].permute(1, 2, 0), cmap="gray")
            plt.title("{}".format(labels[i]), fontsize=27)

        plt.show()
        break