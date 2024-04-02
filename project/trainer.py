import time
import matplotlib.pyplot as plt
from torch import nn, optim, max, save, no_grad


def train(model, train_set, optimizer, loss_function):
    model.train()

    losses = 0.0
    correct = 0
    total = 0

    for images, labels in train_set:
        output, _ = model(images)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        _, predict = max(output.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()

    losses /= len(train_set)
    accuracy = (correct / total) * 100

    return losses, accuracy


def validate(model, val_set, loss_function):
    model.eval()

    losses = 0.0
    correct = 0
    total = 0

    with no_grad():
        for images, labels in val_set:
            output, _ = model(images)
            loss = loss_function(output, labels)
            losses += loss.item()
            _, predict = max(output.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    losses /= len(val_set)
    accuracy = (correct / total) * 100

    return losses, accuracy


def plot(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, "-", label="Train Loss")
    plt.plot(val_loss, "-", label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, "-", label="Train Accuracy")
    plt.plot(val_acc, "-", label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def train_model(epochs, model, train_set, val_set, path, lr):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best = -1

    time_start = time.time()

    for epoch in range(epochs):
        print("EPOCH [{}/{}]".format(epoch + 1, epochs))

        loss, accuracy = train(model, train_set, optimizer, loss_function)
        train_loss.append(loss)
        train_acc.append(accuracy)
        print("TRAIN: Loss [{:.3f}] | Accuracy [{:.2f}%]".format(loss, accuracy))

        loss, accuracy = validate(model, val_set, loss_function)
        val_loss.append(loss)
        val_acc.append(accuracy)
        print("VALIDATION: Loss [{:.3f}] | Accuracy [{:.2f}%]\n".format(loss, accuracy))

        if accuracy > best:
            best = accuracy
            save(model.state_dict(), f"{path}/best.pth")

        save(model.state_dict(), f"{path}/epoch_{epoch + 1:02}.pth")

    time_elapsed = time.time() - time_start
    save(model.state_dict(), f"{path}/model.pth")
    plot(train_loss, val_loss, train_acc, val_acc)
    print("\nTraining complete in {:.1f} seconds\n".format(time_elapsed))