import os
import pickle
import shutil
from scipy import stats
from random import choice
import umap.umap_ as umap
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from torch import no_grad, tensor, softmax
from torchvision.transforms import ToTensor, v2
from sklearn.model_selection import train_test_split


def save_mnist(path, batch_size):
    file = os.path.join(path, "mnist_dataset.pkl")

    if not os.path.exists(file):
        train = MNIST(root="download", train=True, download=True, transform=ToTensor())
        test = MNIST(root="download", train=False, download=True, transform=ToTensor())
        shutil.rmtree("download")

        train, val = train_test_split(train, train_size=50000, test_size=10000, random_state=13)

        mnist = {
            "train": DataLoader(train, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test, batch_size=batch_size, shuffle=False),
        }

        with open(file, "wb") as handle:
            pickle.dump(mnist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_augmented(path, batch_size):
    file = os.path.join(path, "augmented_dataset.pkl")

    if not os.path.exists(file):
        mnist = get_dataset(path, "mnist")
        recrop = v2.RandomResizedCrop(28, (0.50, 0.75))
        rotate = lambda x: v2.RandomRotation(degrees=choice([(-45, -30), (30, 45)]))(x)

        augmented = []

        for image, label in mnist["train"].dataset:
            augmented.append((image, label))

            for _ in range(2):
                augmented.append((recrop(image), label))
                augmented.append((rotate(image), label))

        augmented = DataLoader(augmented, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_classified(path, model, device):
    cuda = device == "cuda"
    file = os.path.join(path, "classified_dataset.pkl")

    if not os.path.exists(file):
        augmented = get_dataset(path, "augmented")
        classified = {}

        model.eval()
        with no_grad():
            for images, labels in augmented:
                if cuda:
                    images = images.cuda()

                logits, embeddings = model(images)

                if cuda:
                    images, logits, embeddings = images.cpu(), logits.cpu(), embeddings.cpu()

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


def save_random(path, batch_size):
    file = os.path.join(path, "random_dataset.pkl")

    if not os.path.exists(file):
        classified = get_dataset(path, "classified")

        random = []
        size = int((sum(len(v) for v in classified.values()) * 0.01) / len(classified))

        for _ in range(size):
            for key in classified.keys():
                index = choice(range(len(classified[key])))
                image, label, _, _ = classified[key].pop(index)
                random.append((image, label))

        random = DataLoader(random, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(random, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_entropy(path, batch_size, reverse):
    order = "max" if reverse else "min"
    file = os.path.join(path, f"{order}_entropy_dataset.pkl")

    if not os.path.exists(file):
        classified = get_dataset(path, "classified")

        for key in classified:
            for i in range(len(classified[key])):
                image, label, logits, embeddings = classified[key][i]
                probabilities = softmax(tensor(logits), dim=0).numpy()
                entropy = stats.entropy(pk=probabilities, base=2)
                classified[key][i] = (entropy, (image, label, logits, embeddings))

            classified[key].sort(key=lambda x: x[0], reverse=reverse)

        entropy = []
        size = int((sum(len(v) for v in classified.values()) * 0.01) / len(classified))

        for _ in range(size):
            for key in classified.keys():
                _, item = classified[key].pop(0)
                image, label, _, _ = item
                entropy.append((image, label))

        entropy = DataLoader(entropy, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(entropy, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_farthest(path, batch_size, metric):
    valid_metrics = ["euclidean", "cosine"]
    assert metric in valid_metrics

    file = os.path.join(path, f"{metric}_dataset.pkl")

    if not os.path.exists(file):
        classified = get_dataset(path, "classified")

        farthest = []
        subset_embeddings = []
        classified_embeddings = {
            0: [item[3] for item in classified[0]],
            1: [item[3] for item in classified[1]],
            2: [item[3] for item in classified[2]],
            3: [item[3] for item in classified[3]],
            4: [item[3] for item in classified[4]],
            5: [item[3] for item in classified[5]],
            6: [item[3] for item in classified[6]],
            7: [item[3] for item in classified[7]],
            8: [item[3] for item in classified[8]],
            9: [item[3] for item in classified[9]],
        }

        size = int((sum(len(v) for v in classified.values()) * 0.01) / len(classified))
        print(f"Metric: {metric} | Progress: 0%")

        for i in range(size):
            if (i + 1) % (size // 10) == 0:
                print(f"Metric: {metric} | Progress: {int(((i + 1) * 100) // size)}%")

            for key in classified.keys():
                if not farthest:
                    item = classified[key].pop(-1)
                    embedding = classified_embeddings[key].pop(-1)
                    image, label, _, _ = item
                    farthest.append((image, label))
                    subset_embeddings.append(embedding)
                else:
                    distances = cdist(subset_embeddings, classified_embeddings[key], metric)

                    if metric == "euclidean":
                        index = distances.sum(axis=0).argmax()
                    elif metric == "cosine":
                        index = distances.sum(axis=0).argmin()

                    item = classified[key].pop(index)
                    embedding = classified_embeddings[key].pop(index)
                    image, label, _, _ = item
                    farthest.append((image, label))
                    subset_embeddings.append(embedding)

        farthest = DataLoader(farthest, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(farthest, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_new_farthest(path, batch_size, model, device, metric):
    valid_metrics = ["euclidean", "cosine"]
    assert metric in valid_metrics

    file = os.path.join(path, f"new_{metric}_dataset.pkl")

    if not os.path.exists(file):
        augmented, full_embeddings, farthest, subset_embeddings, size = get_checkpoint(path, model, device)
        max_size = 2500

        print(f"Metric: {metric} | Progress: {int((size / max_size) * 100)}%")

        for i in range(size, max_size):
            if (i + 1) % (max_size // 10) == 0:
                set_checkpoint(path, augmented, full_embeddings, farthest, subset_embeddings, len(farthest))
                print(f"Metric: {metric} | Progress: {int(((i + 1) / (max_size)) * 100)}%")

            if not farthest:
                image, label = augmented.pop(-1)
                farthest.append((image, label))
                embedding = full_embeddings.pop(-1)
                subset_embeddings.append(embedding)
                continue

            distances = cdist(subset_embeddings, full_embeddings, metric)

            if metric == "euclidean":
                index = distances.sum(axis=0).argmax()
            elif metric == "cosine":
                index = distances.sum(axis=0).argmin()

            augmented[index], augmented[-1] = augmented[-1], augmented[index]
            image, label = augmented.pop(-1)
            farthest.append((image, label))

            full_embeddings[index], full_embeddings[-1] = full_embeddings[-1], full_embeddings[index]
            embedding = full_embeddings.pop(-1)
            subset_embeddings.append(embedding)

        farthest = DataLoader(farthest, batch_size=batch_size, shuffle=True)

        with open(file, "wb") as handle:
            pickle.dump(farthest, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_full_embeddings(path, model, device):
    cuda = device == "cuda"
    file = os.path.join(path, "full_embeddings.pkl")

    if not os.path.exists(file):
        dataset = get_dataset(path, "augmented")
        augmented = []
        full_embeddings = []

        model.eval()
        with no_grad():
            for images, labels in dataset:
                if cuda:
                    images = images.cuda()

                _, embeddings = model(images)

                if cuda:
                    images, embeddings = images.cpu(), embeddings.cpu()

                for i in range(len(images)):
                    image = images[i].numpy()
                    label = labels[i].item()
                    embedding = embeddings[i].numpy()
                    augmented.append((image, label))
                    full_embeddings.append((embedding))

        return augmented, full_embeddings


def get_checkpoint(path, model, device):
    file = os.path.join(path, "checkpoint.pkl")

    if os.path.exists(file):
        with open(file, "rb") as handle:
            checkpoint = pickle.load(handle)

        augmented = checkpoint["augmented"]
        full_embeddings = checkpoint["full_embeddings"]
        farthest = checkpoint["farthest"]
        subset_embeddings = checkpoint["subset_embeddings"]
        size = checkpoint["size"]

        return augmented, full_embeddings, farthest, subset_embeddings, size

    augmented, full_embeddings = get_full_embeddings(path, model, device)
    farthest = []
    subset_embeddings = []
    size = int(0)

    return augmented, full_embeddings, farthest, subset_embeddings, size


def set_checkpoint(path, augmented, full_embeddings, farthest, subset_embeddings, size):
    file = os.path.join(path, "checkpoint.pkl")

    checkpoint = {
        "augmented": augmented,
        "full_embeddings": full_embeddings,
        "farthest": farthest,
        "subset_embeddings": subset_embeddings,
        "size": size,
    }

    with open(file, "wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_base_datasets(data_path, batch_size):
    save_mnist(data_path, batch_size)
    save_augmented(data_path, batch_size)

    files = [
        "mnist_dataset.pkl",
        "augmented_dataset.pkl",
    ]

    for file in files:
        assert os.path.exists(os.path.join(data_path, file))


def save_new_datasets(data_path, model, batch_size, device):
    save_classified(data_path, model, device)
    save_random(data_path, batch_size)
    save_entropy(data_path, batch_size, True)
    save_entropy(data_path, batch_size, False)
    save_farthest(data_path, batch_size, "euclidean")
    save_farthest(data_path, batch_size, "cosine")
    save_new_farthest(data_path, batch_size, model, device, "euclidean")
    # save_new_farthest(data_path, batch_size, model, device, "cosine")

    files = [
        "classified_dataset.pkl",
        "random_dataset.pkl",
        "max_entropy_dataset.pkl",
        "min_entropy_dataset.pkl",
        "euclidean_dataset.pkl",
        "cosine_dataset.pkl",
        "new_euclidean_dataset.pkl",
        # "new_cosine_dataset.pkl",
    ]

    for file in files:
        assert os.path.exists(os.path.join(data_path, file))


def get_dataset(path, target):
    file = os.path.join(path, f"{target}_dataset.pkl")
    assert os.path.exists(file)

    with open(file, "rb") as handle:
        return pickle.load(handle)


def plot_batch(dataset, predicts):
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

            title = "{}".format(labels[i])
            fontsize = 27
            color = "black"

            if predicts is not None:
                if predicts[i] == labels[i]:
                    title = "{}".format(predicts[i])
                    color = "green"
                else:
                    title = "{} (correct: {})".format(predicts[i], labels[i])
                    color = "red"

            plt.title(title, fontsize=fontsize, color=color)

        plt.show()
        break


def save_embeddings(model, target, dataset, data_path, device):
    cuda = device == "cuda"
    file = os.path.join(data_path, f"{target}_embeddings.pkl")

    if not os.path.exists(file):
        data = []

        model.eval()
        with no_grad():
            for images, labels in dataset:
                if cuda:
                    images = images.cuda()

                _, embeddings = model(images)

                if cuda:
                    embeddings = embeddings.cpu()

                for i in range(len(images)):
                    embedding = embeddings[i].numpy()
                    label = labels[i].item()
                    entry = {"embedding": embedding, "label": label}
                    data.append(entry)

        with open(file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_embeddings(target, data_path):
    file = os.path.join(data_path, f"{target}_embeddings.pkl")

    with open(file, "rb") as handle:
        data = pickle.load(handle)

    embeddings = [entry["embedding"] for entry in data[:1000]]
    labels = [entry["label"] for entry in data[:1000]]
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(embeddings)

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10")
    plt.colorbar()
    plt.title("MNIST")
    plt.show()


def study_entropy(path, reverse):
    classified = get_dataset(path, "classified")

    for key in classified:
        for i in range(len(classified[key])):
            image, label, logits, embeddings = classified[key][i]
            probabilities = softmax(tensor(logits), dim=0).numpy()
            entropy = stats.entropy(pk=probabilities, base=2)
            classified[key][i] = (entropy, (image, label, logits, embeddings))

        classified[key].sort(key=lambda x: x[0], reverse=reverse)

    for key in classified:
        entropy, item = classified[key][0]
        image, label, logits, _ = item
        softmaxs = softmax(tensor(logits.tolist()), dim=0).numpy().tolist()

        plt.imshow(image[0].squeeze(), cmap="gray")
        plt.title(label)
        plt.show()

        print(f"\nLogits: {logits.tolist()}")
        print(f"Softmax: {softmaxs}")
        print(f"Entropy: {entropy}\n")

        for i, probability in enumerate(softmaxs):
            print(f"Label {i} probability: {(probability * 100):.2f}%")

        print()