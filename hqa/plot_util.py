import matplotlib.pyplot as plt
import torch

from util import mu_law_decoding, wav_to_float

plt.switch_backend("agg")


def reconstruct_for_tensorboard(x_mu, model):
    """
    Reconstruct all the way to floats in range -1, 1
    """
    model.eval()
    recon_mu = model.reconstruct(x_mu)
    x_mu = x_mu.squeeze(1)
    recon_mu = recon_mu.squeeze(1)

    x_float = wav_to_float(mu_law_decoding(x_mu)).cpu().numpy()
    recon_float = wav_to_float(mu_law_decoding(recon_mu)).cpu().numpy()

    model.train()
    if model.prev_model is not None:
        model.prev_model.eval()

    return x_float, recon_float


def plot_l2_distances(x):
    """ Plot pairwise euclidean distance between rows of input tensor """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(torch.cdist(x.detach().cpu(), x.detach().cpu(), p=2), vmax=50)
    fig.colorbar(im, fraction=0.01, pad=0.02)
    return fig


def plot_histogram(x, bins):
    """ Plot histogram over frequency bins """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(x.detach().cpu().flatten(), bins=bins)
    return fig
