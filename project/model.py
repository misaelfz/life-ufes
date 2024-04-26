import os
import time
import torch
import data as dt
import matplotlib.pyplot as plt
from torch import nn, optim, max, save, no_grad


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.leaky_relu(self.fc2(x))
        embeddings = self.dropout2(x)
        logits = self.fc3(embeddings)
        return logits, embeddings


def train_model(epochs, model, dataset, train_set, val_set, path, lr, wd, device):
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
    loss_function = nn.CrossEntropyLoss()
    cuda = device == "cuda"

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best = -1

    time_start = time.time()

    for epoch in range(epochs):
        print("EPOCH [{}/{}]".format(epoch + 1, epochs))

        loss, accuracy = train_step(model, train_set, optimizer, loss_function, cuda)
        train_loss.append(loss)
        train_acc.append(accuracy)
        print("TRAIN: Loss [{:.3f}] | Accuracy [{:.2f}%]".format(loss, accuracy))

        loss, accuracy = validate_step(model, val_set, loss_function, cuda)
        val_loss.append(loss)
        val_acc.append(accuracy)
        print("VALIDATION: Loss [{:.3f}] | Accuracy [{:.2f}%]\n".format(loss, accuracy))

        if accuracy > best:
            best = accuracy
            save(model.state_dict(), f"{path}/{dataset}_best.pth")

        # save(model.state_dict(), f"{path}/epoch{epoch + 1:02}.pth")

    time_elapsed = time.time() - time_start
    # save(model.state_dict(), f"{path}/model.pth")
    plot_train(train_loss, val_loss, train_acc, val_acc)
    print("\nTraining complete in {:.1f} seconds\n".format(time_elapsed))


def train_step(model, train_set, optimizer, loss_function, cuda):
    losses = 0.0
    correct = 0
    total = 0

    model.train()
    for images, labels in train_set:
        if cuda:
            images, labels = images.cuda(), labels.cuda()

        logits, _ = model(images)
        loss = loss_function(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        _, predicts = max(logits.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum().item()

    losses /= len(train_set)
    accuracy = (correct / total) * 100

    return losses, accuracy


def validate_step(model, val_set, loss_function, cuda):
    losses = 0.0
    correct = 0
    total = 0

    model.eval()
    with no_grad():
        for images, labels in val_set:
            if cuda:
                images, labels = images.cuda(), labels.cuda()

            logits, _ = model(images)
            loss = loss_function(logits, labels)
            losses += loss.item()
            _, predicts = max(logits.data, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()

    losses /= len(val_set)
    accuracy = (correct / total) * 100

    return losses, accuracy


def plot_train(train_loss, val_loss, train_acc, val_acc):
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


def test_model(model, test_set, device):
    cuda = device == "cuda"

    correct = 0.0
    total = 0.0
    first = True

    model.eval()
    with no_grad():
        for images, labels in test_set:
            if cuda:
                images, labels = images.cuda(), labels.cuda()

            logits, _ = model(images)
            _, predicts = max(logits.data, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()

            if first:
                dt.plot_batch(test_set, predicts)
                first = False

    print("\nAccuracy: {:.2f}%".format((correct / total) * 100))


def load_model(model, model_path, target, device):
    model_load = os.path.join(model_path, f"{target}_best.pth")
    model.load_state_dict(torch.load(model_load, map_location=torch.device(device)))

    return model