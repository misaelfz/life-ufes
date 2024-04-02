from torch import max, no_grad
import matplotlib.pyplot as plt


def plot_predicts(images, labels, predict):
    num_rows = 100
    num_cols = 10

    plt.figure(figsize=(40, 4 * num_rows))

    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
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


def test_model(model, dataset):
    model.eval()

    correct = 0.0
    total = 0.0
    plotter = True

    with no_grad():
        for images, labels in dataset["test"]:
            output, _ = model(images)
            _, predict = max(output.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            if plotter:
                plot_predicts(images, labels, predict)
                plotter = False

    print("\nAccuracy: {:.2f}%".format((correct / total) * 100))