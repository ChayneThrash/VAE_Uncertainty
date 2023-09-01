# This is a sample Python script.
import torchvision.transforms
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from torchvision.transforms.functional import pil_to_tensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from VAE import VAE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import seaborn as sns
from operator import attrgetter


def train(train_ds, model, epochs, device, optimizer, batch_size):
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    epoch_stats = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kld_loss = 0.0
        n_samples = 0
        model.train()
        for x, y in tqdm(train_dl, leave=False):
            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            logits, mu, log_var = model(x)
            loss, class_loss, kld_loss = model.loss_function(logits, y, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            total_ce_loss += class_loss.item()
            total_kld_loss += kld_loss.item()
            n_samples += x.size()[0]
            optimizer.step()

        average_train_loss = total_loss / float(n_samples)
        average_ce_loss = total_ce_loss / float(n_samples)
        average_kld_loss = total_kld_loss / float(n_samples)

        epoch_stats.append((epoch, average_train_loss, average_ce_loss, average_kld_loss))
        if epoch % 1 == 0:
            print("Epoch {0} finished. Train loss: {1}, CE loss: {2}, KLD loss: {3}"
                  .format(epoch, average_train_loss, average_ce_loss, average_kld_loss))

    return epoch_stats


def eval_model(model, ds, device):
    with torch.no_grad():
        model.eval()
        total_correct = 0
        for index, (x, y) in enumerate(ds):
            x = x.to(device)
            logits, _, _ = model(x)
            pred = torch.argmax(logits).item()
            if pred == y:
                total_correct += 1
        return total_correct / len(ds)


def bucket_incorrect_samples(model, test_ds, device):
    with torch.no_grad():
        model.eval()
        samples = []
        for test_index, (x, y) in enumerate(test_ds):
            x = x.to(device)
            logits, mu, log_var = model(x)
            var = (0.5 * log_var).exp()
            pred = torch.argmax(logits).item()
            pred_scores = torch.nn.functional.softmax(logits)
            confidence = pred_scores[pred]
            samples.append(dict(sample=x, var=var, confidence=confidence, label=y, pred=pred,
                                          test_index=test_index))
        buckets = {"98%+": [], "95%+": [], "90%+": [], "85%+": [], "80%+": [], "70%+": [], "other (incorrect)": [],
                   "other (correct)": []}
        for sample in samples:
            if sample["label"] != sample["pred"]:
                if sample["confidence"] >= 0.98:
                    buckets["98%+"].append(sample)
                if sample["confidence"] >= 0.95:
                    buckets["95%+"].append(sample)
                elif sample["confidence"] >= 0.9:
                    buckets["90%+"].append(sample)
                elif sample["confidence"] >= 0.85:
                    buckets["85%+"].append(sample)
                elif sample["confidence"] >= 0.80:
                    buckets["80%+"].append(sample)
                elif sample["confidence"] >= 0.7:
                    buckets["70%+"].append(sample)
                else:
                    buckets["other (incorrect)"].append(sample)
            else:
                buckets["other (correct)"].append(sample)
        return buckets


def visualize_buckets(buckets):
    for bucket in ["95%+", "90%+", "85%+"]:
        for i in range(0, 3):
            sample_info = buckets[bucket][i]
            plt.imshow(sample_info["sample"].view((28, 28)), cmap='gray')
            plt.title("label={0}, pred={1}, conf={2}".format(sample_info["label"], sample_info["pred"],
                                                             sample_info["confidence"]))
            plt.show()
    print("")


def visualize_latent_space(model, ds, device):
    model.eval()
    with torch.no_grad():
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = 10

        mus = []
        labels = []
        for x, y in ds:
            x = x.to(device)
            mu, _ = model.encode(x)
            mus.append(mu.numpy())
            labels.append(y)

        mus = np.array(mus)
        labels = np.array(labels)
        cmap = cm.get_cmap('tab20')
        for lab in range(num_categories):
            indices = labels == lab
            ax.scatter(mus[indices, 0], mus[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                       alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.show()

def load_mnist():
    transforms = torchvision.transforms.Compose([
        pil_to_tensor,
        lambda x: x.squeeze().flatten().float()
    ])

    mnist_train = MNIST(root="./data", train=True, download=True, transform=transforms)
    mnist_test = MNIST(root="./data", train=False, download=False, transform=transforms)

    return mnist_train, mnist_test


def train_vae(mnist_train, model):
    device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(mnist_train, model, 30, device, optimizer, 1024)
    torch.save(model.state_dict(), './model.pt')
    torch.save(optimizer.state_dict(), './optim.pt')

def plot_variance_stats(buckets):
    bucket_mean = {"percentile": [], "mean_variance": []}
    bucket_median = {"percentile": [], "median_variance": []}
    for bucket in buckets:
        samples = buckets[bucket]
        max_var = np.array([s["var"].max() for s in samples])
        mean_max_var = max_var.mean()
        median_max_var = np.median(max_var)
        bucket_mean["percentile"].append(bucket)
        bucket_mean["mean_variance"].append(mean_max_var)
        bucket_median["percentile"].append(bucket)
        bucket_median["median_variance"].append(median_max_var)
    sns.set_theme()
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(bucket_mean, x="percentile", y="mean_variance")
    plt.title("mean variance")
    plt.show()
    sns.barplot(bucket_median, x="percentile", y="median_variance", )
    plt.title("median variance")
    plt.show()


if __name__ == '__main__':
    device = torch.device("cpu")
    model = VAE(in_dim=784, hidden_dim=64, latent_dim=8, output_dim=10)
    mnist_train, mnist_test = load_mnist()
    # train_vae(mnist_train, model)
    model.load_state_dict(torch.load('./model.pt'))

    for test_index in [7472, 4823, 6574]:
        image = mnist_test[test_index][0].view((28, 28))
        label = mnist_test[test_index][1]
        plt.imshow(image, cmap='gray')
        plt.title("label={0}".format(label))
    plt.show()


    # print("test accuracy: {0}".format(eval_model(model, mnist_test, device)))
    # buckets = bucket_incorrect_samples(model, mnist_test, device)

    # for percentile in ["95%+", "90%+", "85%+"]:
    #     bucket = buckets[percentile]
    #     bucket.sort(reverse=True, key=attrgetter('confidence'))

    # plot_variance_stats(buckets)
    # print("hi")
    #visualize_latent_space(model, mnist_test, device)
