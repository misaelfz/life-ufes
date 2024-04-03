from torch import max, no_grad
import matplotlib.pyplot as plt


def test_model(model, mnist, device):
    model.eval()
    cuda = device == "cuda"
    batch_size = int(len(mnist["train"].dataset) / len(mnist["train"]))

    correct = 0.0
    total = 0.0
    plot = True

    with no_grad():
        for images, labels in mnist["test"]:
            if cuda:
                images, labels = images.cuda(), labels.cuda()

            output, _ = model(images)
            _, predict = max(output.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            if plot:
                plot_predicts(images.to("cpu"), labels.to("cpu"), predict, batch_size)
                plot = False

    print("\nAccuracy: {:.2f}%".format((correct / total) * 100))


def plot_predicts(images, labels, predict, batch_size):
    rows = int(batch_size / 10)
    cols = int(10)

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