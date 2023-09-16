# This is a sample Python script.
import PIL.Image
import seaborn
import torchvision.transforms
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from torchvision.transforms.functional import pil_to_tensor
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from VAE import VAE
from VAE_covariance import VAE_covariance
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import seaborn as sns
from operator import itemgetter
from MixupDataset import MixupDataset
from utils import create_cholesky_matrix, aggregate_variance, get_var_logits_entropy, entropy
import os
import pandas as pd
from wasserstein import calc_wasserstein_barycenter
from plotting import draw_conf2D


def train(train_ds, model, max_epochs, device, optimizer, batch_size):
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    epoch_stats = []

    previous_loss = 0.0
    for epoch in range(max_epochs):
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kld_loss = 0.0
        n_samples = 0
        model.train()
        for x, y in tqdm(train_dl, leave=False):
            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            logits, mu, log_var, upper_L = model(x)
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

        change_in_loss = abs(previous_loss - average_train_loss)
        epoch_stats.append((epoch, average_train_loss, average_ce_loss, average_kld_loss))
        if epoch % 1 == 0:
            print("Epoch {0} finished. Train loss: {1}, CE loss: {2}, KLD loss: {3}, Loss change: {4}"
                  .format(epoch, average_train_loss, average_ce_loss, average_kld_loss, change_in_loss))

        if change_in_loss <= 1e-6:
            break
        previous_loss = average_train_loss

    return epoch_stats


def eval_model(model, ds, device):
    with torch.no_grad():
        model.eval()
        total_correct = 0
        for index, (x, y) in enumerate(ds):
            x = x.to(device)
            logits, _, _, _ = model.inference(x)
            pred = torch.argmax(logits).item()
            if pred == y:
                total_correct += 1
        return total_correct / len(ds)


def create_sample_result(model, device, x, y):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits, mu, log_var, upper_L = model.inference(x)
        if upper_L is None:
            var = (0.5 * log_var).exp()
        else:
            L = create_cholesky_matrix(log_var, upper_L)
            var = torch.linalg.eigvals(torch.bmm(L, torch.permute(L, (0, 2, 1)))).real
            var = torch.squeeze(var)
        pred = torch.argmax(logits).item()
        pred_scores = torch.squeeze(torch.nn.functional.softmax(logits))
        confidence = pred_scores[pred]
        return dict(sample=x, var=var, confidence=confidence, label=y, pred=pred, pred_scores=pred_scores)


def bucket_samples(model, test_ds, device):
    with torch.no_grad():
        model.eval()
        samples = []
        for test_index, (x, y) in enumerate(test_ds):
            samples.append(create_sample_result(model, device, x, y))

        buckets = {"98%+": [], "95%+": [], "90%+": [], "85%+": [], "80%+": [], "70%+": [], "other (incorrect)": [],
                   "other (correct)": []}
        for sample in samples:
            if sample["label"] != sample["pred"]:
                if sample["confidence"] >= 0.98:
                    buckets["98%+"].append(sample)
                elif sample["confidence"] >= 0.95:
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


def load_mnist(flatten=False):
    transforms = [pil_to_tensor]
    if flatten:
        transforms.append(lambda x: x.squeeze().flatten())
    transforms.append(lambda x: x.float())
    transforms = torchvision.transforms.Compose(transforms)

    mnist_train = MNIST(root="./data", train=True, download=True, transform=transforms)
    mnist_test = MNIST(root="./data", train=False, download=False, transform=transforms)

    return mnist_train, mnist_test


def load_fashion_mnist(flatten=False):
    transforms = [pil_to_tensor]
    if flatten:
        transforms.append(lambda x: x.squeeze().flatten())
    transforms.append(lambda x: x.float())
    transforms = torchvision.transforms.Compose(transforms)

    fmnist_train = FashionMNIST(root="./data", train=True, download=True, transform=transforms)
    fmnist_test = FashionMNIST(root="./data", train=False, download=False, transform=transforms)

    return fmnist_train,  fmnist_test


def train_vae(mnist_train, model, device, output_path, max_epochs):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(mnist_train, model, max_epochs, device, optimizer, 1024)
    torch.save(model.state_dict(), os.path.join(output_path,  "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_path,  "optim.pt"))


def plot_variance_stats(buckets, output_path, output_name_base, method='max', model=None, device=None):
    bucket_mean = {"percentile": [], "mean_variance": []}
    bucket_median = {"percentile": [], "median_variance": []}
    for bucket in buckets:
        samples = buckets[bucket]
        if method == 'logits_var_entropy':
            agg_var = np.array([get_var_logits_entropy(model, device, s['sample']) for s in samples])
        else:
            agg_var = np.array([aggregate_variance(method, s["var"]) for s in samples])
        mean_max_var = agg_var.mean()
        median_max_var = np.median(agg_var)
        bucket_mean["percentile"].append(bucket)
        bucket_mean["mean_variance"].append(mean_max_var)
        bucket_median["percentile"].append(bucket)
        bucket_median["median_variance"].append(median_max_var)
    plt.figure(figsize=(12, 12), clear=True)
    sns.barplot(bucket_median, x="percentile", y="median_variance")
    plt.title("median variance")
    plt.yscale("log")
    plt.savefig(os.path.join(output_path, "median_" + output_name_base + ".png"))
    plt.close()


def load_ambiguous_image_set(mnist_test_ds):
    with open('results/ambiguous_images/sampled/indices.txt', 'r') as f:
        ambiguous_image_indices = [int(line.strip()) for line in  f.readlines()]

    ambiguous_images = [mnist_test_ds[i][0] for i in ambiguous_image_indices]
    return ambiguous_images


def create_ambiguous_image_set(mnist_ds):
    sample = random.sample(range(len(mnist_ds)), 500)
    saved_indices = []
    for test_index in sample:
        x, y = mnist_ds[test_index]
        plt.imshow(x.view((28, 28)), cmap="gray")
        plt.title("label={0}".format(y))
        plt.show(block=False)
        keypress = input()
        if keypress == 'y':
            saved_indices.append(test_index)
            if len(saved_indices) >= 100:
                break
        plt.close()
    with open('results/ambiguous_images/sampled/indices.txt', "w+") as f:
        f.writelines([str(i) + '\n' for i in saved_indices])


def evaluate_variance_directions(mnist_test, model, device):
    mixup_test = MixupDataset(mnist_test, 1.0)
    n_correct = 0
    n_correct_mix = 0
    for x, y in mnist_test:
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            logits, mu, log_var, upper_L = model.inference(x)
            mu = mu.view(-1)
            L = create_cholesky_matrix(log_var, upper_L)
            var, directions = torch.linalg.eig(torch.bmm(L, torch.permute(L, (0, 2, 1))))
            var = torch.squeeze(var)

            mix_x, mix_y = mixup_test.sample(y)
            mix_x = 0.5 * (x + mix_x)
            logits, mu_mix, log_var, upper_L = model.inference(mix_x)
            mu_mix = mu_mix.view(-1)
            L = create_cholesky_matrix(log_var, upper_L)
            mix_var, mix_directions = torch.linalg.eig(torch.bmm(L, torch.permute(L, (0, 2, 1))))
            mix_var = torch.squeeze(mix_var)
            print("")
        model.zero_grad()
        mu.requires_grad = True
        logits = model.decode(mu).view(-1)
        grads = torch.zeros((10, 8))
        for i in range(0, 10):
            grads[i] = torch.autograd.grad(logits[i], mu, retain_graph=True)[0]

        top_class_dir1 = torch.argmin(torch.cosine_similarity(grads, directions[:, 0].real))
        top_class_dir2 = torch.argmin(torch.cosine_similarity(grads, directions[:, 1].real))

        model.zero_grad()
        mu_mix.requires_grad = True
        logits = model.decode(mu_mix).view(-1)
        for i in range(0, 10):
            grads[i] = torch.autograd.grad(logits[i], mu_mix, retain_graph=True)[0]

        mix_top_class_dir1 = torch.argmin(torch.cosine_similarity(grads, mix_directions[:, 0].real))
        mix_top_class_dir2 = torch.argmin(torch.cosine_similarity(grads, mix_directions[:, 1].real))
        if top_class_dir1 == y:
            n_correct += 1
        if mix_top_class_dir1 == y or mix_top_class_dir1 == mix_y:
            n_correct_mix += 1
    print("# correct: {0}, # correct (mixup): {1}".format(n_correct, n_correct_mix))


def output_high_variance_samples(model, mnist_test, device, output_path):
    high_variance_samples = sample_high_variance_samples(model, mnist_test, device, 20)
    high_variance_samples_path = os.path.join(output_path, 'high_variance_samples')
    if not os.path.exists(high_variance_samples_path):
        os.makedirs(high_variance_samples_path)
    for i, sample in enumerate(high_variance_samples):
        plt.imshow(sample["sample"].view(28, 28), cmap="gray")
        plt.title("label={0}, pred={1}, conf={2}, var={3}".format(sample["label"], sample["pred"],
                                                                  sample["confidence"], sample["agg_var"]))
        plt.savefig(os.path.join(high_variance_samples_path, str(i) + ".png"))
        plt.close()


def sample_high_variance_samples(model, mnist_test, device, k):
    sample_results = []
    for (x, y) in mnist_test:
        sample_result = create_sample_result(model, device, x, y)
        sample_result["agg_var"] = aggregate_variance("norm", sample_result["var"])
        sample_results.append(sample_result)
    sample_results.sort(key=itemgetter("agg_var"), reverse=True)
    return sample_results[:k]


def create_rejection_classification_plot(all_mnist_results, all_fmnist_results, method,
                                         output_dir, plot_title,  model=None, device=None):
    percentiles = [i / 100.0 for i in range(0, 101, 5)]
    if method == 'pred_entropy':
        all_mnist_scores = [entropy(r["pred_scores"]) for r in all_mnist_results]
        all_fmnist_scores = [entropy(r["pred_scores"]) for r in all_fmnist_results]
    elif method == 'logits_var_entropy':
        all_mnist_scores = np.array([get_var_logits_entropy(model, device, s['sample']) for s in all_mnist_results])
        all_fmnist_scores = np.array([get_var_logits_entropy(model, device, s['sample']) for s in all_fmnist_results])
    else:
        all_mnist_scores = np.array([aggregate_variance(method, s["var"]) for s in all_mnist_results])
        all_fmnist_scores = np.array([aggregate_variance(method, s["var"]) for s in all_fmnist_results])
    all_scores = []
    all_scores.extend(all_mnist_scores)
    all_scores.extend(all_fmnist_scores)
    bounds = np.quantile(np.array(all_scores), percentiles)[::-1]
    accuracies = []
    all_mnist_percent_not_rejected = []
    test_accuracy = len([r for r in all_mnist_results if r["label"] == r["pred"]]) / len(all_mnist_results)
    for p, upper_bound in zip(percentiles, bounds):
        total_samples = 0
        total_correct = 0
        for score, sample in zip(all_mnist_scores, all_mnist_results):
            if score <= upper_bound:
                total_samples += 1
                if sample["label"] == sample["pred"]:
                    total_correct += 1
        all_mnist_percent_not_rejected.append(total_samples / len(all_mnist_results))
        for score, sample in zip(all_fmnist_scores, all_fmnist_results):
            if score <= upper_bound:
                total_samples += 1
        accuracies.append(float(total_correct) / float(total_samples))
    test_accuracy_line = [test_accuracy] * len(percentiles)
    d = pd.DataFrame(data={"percentiles": percentiles, "accuracy": accuracies,
                           "percent_mnist": all_mnist_percent_not_rejected,
                           "test_accuracy": test_accuracy_line})
    d.set_index("percentiles")
    g = seaborn.lineplot(x="percentiles", y='value',
                         data=pd.melt(d, ["percentiles"]), hue='variable',
                         markers=['.', 'o', "none"])
    g.set_yticks([i / 100 for i in range(40, 101, 10)])
    plt.title(plot_title)
    plt.xlabel("percent rejected")
    plt.savefig(os.path.join(output_dir, method + "rejection_classification.png"))
    plt.close()


def create_combined_roc_curve_plot(all_mnist_results, all_fmnist_results, methods,
                                         output_dir, plot_title, model=None, device=None):
    fp_rates = [i / 100.0 for i in range(0, 11, 2)]
    fp_rates.extend([i / 100.0 for i in range(15, 101, 5)])
    d = pd.DataFrame(data={"fp_rates": fp_rates})
    d.set_index("fp_rates")
    for method in methods:
        d[method] = create_roc_curve(all_mnist_results, all_fmnist_results, method, fp_rates, model, device)

    g = seaborn.lineplot(x="fp_rates", y='value',
                         data=pd.melt(d, ["fp_rates"]), hue='variable',
                         markers=['.', 'o', "none"])
    plt.title(plot_title)
    plt.xlabel("False Positive Rates")
    plt.ylabel("True Positive Rates")
    plt.savefig(os.path.join(output_dir, "full_ROC.png"))
    plt.close()


def create_roc_curve(all_mnist_results, all_fmnist_results, method, fp_rates, model=None, device=None):
    if method == 'pred_entropy':
        all_mnist_scores = [entropy(r["pred_scores"]) for r in all_mnist_results]
        all_fmnist_scores = [entropy(r["pred_scores"]) for r in all_fmnist_results]
    elif method == 'logits_var_entropy':
        all_mnist_scores = np.array([get_var_logits_entropy(model, device, s['sample']) for s in all_mnist_results])
        all_fmnist_scores = np.array([get_var_logits_entropy(model, device, s['sample']) for s in all_fmnist_results])
    else:
        all_mnist_scores = np.array([aggregate_variance(method, s["var"]) for s in all_mnist_results])
        all_fmnist_scores = np.array([aggregate_variance(method, s["var"]) for s in all_fmnist_results])
    bounds = np.quantile(np.array(all_mnist_scores), fp_rates)[::-1]
    tp_rates = []
    for p, upper_bound in zip(fp_rates, bounds):
        true_positives = 0
        for score, sample in zip(all_fmnist_scores, all_fmnist_results):
            if score > upper_bound:
                true_positives += 1
        tp_rates.append(float(true_positives) / len(all_fmnist_scores))
    return tp_rates


def get_median_variance(method, results):
    scores = np.array([aggregate_variance(method, s["var"]) for s in results])
    return np.median(scores)


def calc_claswise_wasserstein_barycenter(model, train_ds, device, latent_dim):
    train_dl = DataLoader(dataset=train_ds, batch_size=2048, shuffle=True)
    class_bins_mu = []
    class_bins_covar = []
    class_indices = [0]*10

    class_counts = [0]*10
    for x, y in train_ds:
        class_counts[y] += 1

    for i in range(10):
        class_bins_mu.append(np.zeros((class_counts[i], latent_dim)))
        class_bins_covar.append(np.zeros((class_counts[i], latent_dim, latent_dim)))
    with torch.no_grad():
        model.eval()
        for x, y in train_dl:
            x = x.to(device)
            mu, log_var, upper_L = model.encode(x)
            L = create_cholesky_matrix(log_var, upper_L)
            covariance = L @ torch.transpose(L, 1, 2)

            for i, y_val in enumerate(y):
                index = class_indices[y_val]
                class_bins_mu[y_val][index] = mu[i].detach().numpy()
                class_bins_covar[y_val][index] = covariance[i].detach().numpy()
                class_indices[y_val] += 1

    classwise_barycenter_mu = []
    classwise_barycenter_covar = []
    for i in range(10):
        class_mu, class_covar = calc_wasserstein_barycenter(class_bins_mu[i], class_bins_covar[i])
        classwise_barycenter_mu.append(class_mu)
        classwise_barycenter_covar.append(class_covar)
    return classwise_barycenter_mu, classwise_barycenter_covar


def run_experiments(id_set, ood_set, experiment_dir):
    latent_dim = 2
    device = torch.device("cpu")
    output_path = experiment_dir

    id_train, id_test = id_set
    ood_train, ood_test = ood_set

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # model = VAE(in_dim=784, hidden_dim=64, latent_dim=latent_dim, output_dim=10)
    model = VAE_covariance(in_dim=(28, 28), hidden_dim=64, latent_dim=latent_dim, output_dim=10)
    # mnist_train = MixupDataset(mnist_train, 0.5)
    train_vae(id_train, model, device, output_path=output_path, max_epochs=100)
    # id_train_mix = MixupDataset(id_train, 0.5)
    # for param in model.encoder_mu.parameters():
    #     param.requires_grad = False
    # for param in model.decoder.parameters():
    #     param.requires_grad = False
    # train_vae(id_train_mix, model, device, output_path=output_path, max_epochs=10)
    model.load_state_dict(torch.load(os.path.join(output_path, 'model.pt'), map_location=torch.device('cpu')))

    mu_barycenter, covar_barycenter = calc_claswise_wasserstein_barycenter(model, id_train, device, latent_dim)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(10):
        draw_conf2D(mu_barycenter[i], covar_barycenter[i], ax)
    # draw_conf2D(mu, cov, ax, edgecolor='r')
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    # with torch.no_grad():
    #     model.eval()
    #     sampled_x, sampled_y = iter(DataLoader(dataset=id_train, batch_size=, shuffle=True))
    #     sampled_x = sampled_x.to(device)
    #     sampled_mu, sampled_log_var, sampled_upper_L = model.encode(sampled_x)
    #     L = create_cholesky_matrix(sampled_log_var, sampled_upper_L)
    #     covariance = L @ torch.transpose(L, 1, 2)



    # evaluate_variance_directions(id_test, model)

    # print("Accuracy: {0}".format(eval_model(model, id_test, device)))
    # print("Accuracy: {0}".format(eval_model(model, ood_test, device)))
    # buckets = bucket_samples(model, id_test, device)
    # ambiguous_images = load_ambiguous_image_set(id_test)
    # ambiguous_sample_results = []
    # for x in ambiguous_images:
    #     ambiguous_sample_results.append(create_sample_result(model, device, x, None))

    # total_sample_results = []
    # for bucket in buckets:
    #     total_sample_results.extend(buckets[bucket])
    # ambiguous_buckets = {"ambiguous": ambiguous_sample_results, "total": total_sample_results}

    # ood_sample_results = []
    # for x, y in ood_test:
    #     ood_sample_results.append(create_sample_result(model, device, x, None))
    # fmnist_buckets = {"ood": ood_sample_results, "id": total_sample_results}


    # create_rejection_classification_plot(fmnist_buckets["id"], fmnist_buckets["ood"], method="pred_entropy",
    #                                      output_dir=output_path,
    #                                      plot_title="Classification Entropy Rejection Classification")
    # create_rejection_classification_plot(fmnist_buckets["id"], fmnist_buckets["ood"], method="norm",
    #                                      output_dir=output_path, plot_title="Variance Norm Rejection Classification")
    # create_rejection_classification_plot(fmnist_buckets["id"], fmnist_buckets["ood"], method="entropy",
    #                                      output_dir=output_path, plot_title="Variance Entropy Rejection Classification")
    # create_rejection_classification_plot(fmnist_buckets["id"], fmnist_buckets["ood"], method="max",
    #                                      output_dir=output_path, plot_title="Max Variance Rejection Classification")

    # methods = ["pred_entropy", "norm", "entropy", "neg_entropy"]
    # with open(os.path.join(output_path, 'median_variance.txt'), "w+") as f:
    #     for m in methods:
    #         if m == "pred_entropy" or m == 'neg_entropy':
    #             continue
    #         get_median_variance(m, fmnist_buckets['id'])
    #         f.write("ID Method: {0}, Var: {1}\n".format(m, get_median_variance(m, fmnist_buckets['id'])))
    #         f.write("OOD Method: {0}, Var: {1}\n".format(m, get_median_variance(m, fmnist_buckets['ood'])))
    # create_combined_roc_curve_plot(fmnist_buckets["id"], fmnist_buckets["ood"],
    #                                methods=methods, output_dir=output_path,
    #                                plot_title="ROC Curve with different exclusion methods")

    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_norm', method='norm')
    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_max', method='max')
    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_mult', method='mult')
    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_sum', method='sum')
    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_entropy', method='entropy')
    # plot_variance_stats(buckets, output_path, output_name_base='bucketed_logits_var_entropy', method='logits_var_entropy',
    #                     model=model, device=device)
    #
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_norm', method='norm')
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_max', method='max')
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_mult', method='mult')
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_sum', method='sum')
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_entropy', method='entropy')
    # plot_variance_stats(ambiguous_buckets, output_path, output_name_base='ambiguous_logits_var_entropy', method='logits_var_entropy',
    #                     model=model, device=device)
    #
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_norm', method='norm')
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_max', method='max')
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_mult', method='mult')
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_sum', method='sum')
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_entropy', method='entropy')
    # plot_variance_stats(fmnist_buckets, output_path, output_name_base='fmnist_logits_var_entropy',
    #                     method='logits_var_entropy', model=model, device=device)

if __name__ == '__main__':
    sns.set_theme()
    mnist_train, mnist_test = load_mnist()
    fmnist_train, fmnist_test = load_fashion_mnist()

    mnist_experiment_dirs = [
        #'experiment_1/8d_conv'#, 'experiment_2/8d_conv_mixup_0.5', 'experiment_3/8d_conv'
        # 'experiment_4/8d_conv'
        'experiment_1/2d_conv'
    ]

    fmnist_experiment_dirs = [
        # 'experiment_1/8d_conv', 'experiment_2/8d_conv_mixup_0.5', 'experiment_3/8d_conv',
        # 'experiment_4/8d_conv'
        'experiment_1/2d_conv'
    ]

    for experiment_dir in mnist_experiment_dirs:
        run_experiments((mnist_train, mnist_test), (fmnist_train, fmnist_test),
                        experiment_dir=os.path.join('results/mnist', experiment_dir))

    for experiment_dir in fmnist_experiment_dirs:
        run_experiments((fmnist_train, fmnist_test), (mnist_train, mnist_test),
                        experiment_dir=os.path.join('results/fmnist', experiment_dir))
