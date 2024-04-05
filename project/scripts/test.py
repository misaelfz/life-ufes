import os
import pickle
import umap.umap_ as umap
from torch import max, no_grad
import matplotlib.pyplot as plt


def test_model(model, test_set, device):
    cuda = device == "cuda"

    correct = 0.0
    total = 0.0
    plot = True

    model.eval()
    with no_grad():
        for images, labels in test_set:
            if cuda:
                images, labels = images.cuda(), labels.cuda()

            logits, _ = model(images)
            _, predict = max(logits.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            if plot:
                plot_predicts(images.to("cpu"), labels.to("cpu"), predict)
                plot = False

    print("\nAccuracy: {:.2f}%".format((correct / total) * 100))


def plot_predicts(images, labels, predict):
    rows = 10
    cols = 10

    plt.figure(figsize=(40, 4 * rows))

    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].permute(1, 2, 0), cmap="gray")

        if predict[i] == labels[i]:
            plt.title("{}".format(predict[i]), fontsize=27, color=("green"))
        else:
            plt.title(
                "{} (correct: {})".format(predict[i], labels[i]),
                fontsize=27,
                color=("red"),
            )

    plt.show()


def plot_embeddings(model, test_set, data_path, device):
    cuda = device == "cuda"

    file = os.path.join(data_path, "embeddings.pkl")

    if not os.path.exists(file):
        data = []

        model.eval()
        with no_grad():
            for images, labels in test_set:
                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                _, embeddings = model(images)

                for i in range(len(images)):
                    image = images[i].numpy()
                    label = labels[i].item()
                    embedding = embeddings[i].numpy()
                    entry = {"image": image, "label": label, "embedding": embedding}
                    data.append(entry)

        with open(file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
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