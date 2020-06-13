from scipy.stats import norm

import numpy as np
import torch

from util import device


def get_downsample(model, window_size=16384):
    e_in = torch.ones([1, 1, window_size]).to(device)
    e_out = model.encode(e_in)
    return e_in.shape[2] / e_out.shape[2]


def encode_lower(model, x):
    """identical to method in HQA but tracks gradients"""
    if model.prev_model is None:
        return x
    else:
        z_e_lower = encode(model.prev_model, x)
        return z_e_lower


def encode(model, x):
    """identical to method in HQA but tracks gradients"""
    z_e_lower = encode_lower(model, x)
    z_e = model.encoder(z_e_lower)
    return z_e


def decode(model, z_q):
    """identical to method in HQA but tracks gradients and doesn't quantize"""
    if model.prev_model is not None:
        recon = decode(model.prev_model, model.decoder(z_q))
    else:
        recon = model.decoder(z_q)
    return recon


def gaussian_smooth(x_abs, nsig=3, kernlen=21):
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern = np.diff(norm.cdf(x))
    x_smooth = np.convolve(x_abs, kern, mode="same")
    return x_smooth / x_smooth.max()


def get_module_receptive_field(x_in, x_out):
    grads = torch.zeros(x_out.shape).to(device)
    grads[:, :, x_out.shape[2] // 2] = 1  # grad only in center
    x_out.backward(grads)
    x_abs = x_in.grad.cpu().abs().sum(1).squeeze().numpy()  # take absolute value
    kernlen = (x_abs / x_abs.max() > 0.05).sum() // 4
    x_smooth = gaussian_smooth(x_abs, nsig=3, kernlen=kernlen)

    # Find width of mode --> receptive field
    # Threshold is required as gradients aren't always zero away from the mode
    # One reason for this is due to group norm in the encoder
    thresh = x_smooth[: x_smooth.shape[0] // 10].max() * 2 + 0.001
    ind = np.argwhere((x_smooth > thresh)).squeeze()
    rf = ind[-1] - ind[0]
    return rf


def get_latent_receptive_field(layer, num, shape, id_=2):
    d_in = torch.zeros(shape).to(device)
    y_empty = decode(layer, d_in)

    ids = torch.tensor(id_).long().view(1, 1, 1).to(device)
    groups = layer.codebook.codebook_groups
    d_in[:, :, shape[2] // 2] = layer.codebook.lookup(ids).squeeze().repeat(groups)
    y_pulse = decode(layer, d_in)

    # take difference, absolute value, normalise and smooth
    y = (y_pulse - y_empty).abs().cpu()
    x_abs = (y / y.max()).squeeze().detach().numpy()
    kernlen = (x_abs / x_abs.max() > 0.05).sum() // 4
    x_smooth = gaussian_smooth(x_abs, nsig=3, kernlen=kernlen)

    # find width of mode
    thresh = x_smooth[: x_smooth.shape[0] // 10].max() * 2 + 0.001
    ind = np.argwhere((x_smooth > thresh * (num + 1))).squeeze()
    rf = ind[-1] - ind[0]
    return rf


def get_receptive_fields(model, window_size=16384):
    enc_lrf = []
    enc_grf = []
    dec_lrf = []
    dec_grf = []

    e_in = torch.ones([1, 1, window_size], requires_grad=True, device=device)
    for num, layer in enumerate(model):

        # LOCAL ENCODER
        # pass x through encoder of layer and get local receptive field
        e_out = layer.encoder(e_in)
        rfl = get_module_receptive_field(e_in, e_out)
        enc_lrf.append(rfl)
        # initialise x with shape of x_out for next layer
        e_in = torch.ones(e_out.shape, requires_grad=True, device=device)

        # GLOBAL ENCODE
        # pass audio sample domain x through whole stack and get global receptive field
        xg = torch.ones([1, 1, window_size], requires_grad=True, device=device)
        x_out = encode(layer, xg)
        rfg = get_module_receptive_field(xg, x_out)
        enc_grf.append(rfg)

        # LOCAL DECODER
        d_in = torch.ones(e_out.shape, requires_grad=True, device=device)
        d_out = layer.decoder(d_in)
        rfl = get_module_receptive_field(d_in, d_out)
        dec_lrf.append(rfl)

        # GLOBAL DECODE
        xg = torch.ones([1, 1, window_size], requires_grad=True, device=device)
        x_out = decode(layer, encode(layer, xg))
        rfg = get_module_receptive_field(xg, x_out)
        dec_grf.append(rfg)

    return enc_lrf, enc_grf, dec_lrf, dec_grf
