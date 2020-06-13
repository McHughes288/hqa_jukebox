from math import ceil
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, Categorical

from util import Mish
from body.body_wrapper import TrainedBody
from data.streaming import RawStream
from util import mu_law_encoding, mu_law_decoding


class HQA(nn.Module):
    def __init__(self, prev_model, encoder, codebook, decoder, inp_normalizer=None):
        super().__init__()
        self.prev_model = prev_model

        self.encoder = encoder
        self.codebook = codebook
        self.decoder = decoder
        self.inp_normalizer = inp_normalizer

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.codebook, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param

    def forward(self, inp):
        z_e_lower = self.encode_lower(inp)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        return z_e_lower_tilde, z_e_lower, z_q, z_e, indices, kl, commit_loss

    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            with torch.no_grad():
                z_e_lower = self.prev_model.encode(x)
                if self.inp_normalizer is not None:
                    z_e_lower = z_e_lower.transpose(1, 2)
                    z_e_lower = self.inp_normalizer(z_e_lower)
                    z_e_lower = z_e_lower.transpose(1, 2)
            return z_e_lower

    def encode(self, x):
        with torch.no_grad():
            z_e_lower = self.encode_lower(x)
            z_e = self.encoder(z_e_lower)
        return z_e

    def decode_lower(self, z_q_lower):
        with torch.no_grad():
            recon = self.prev_model.decode(z_q_lower)
        return recon

    def decode(self, z_q):
        with torch.no_grad():
            if self.prev_model is not None:
                z_e_lower = self.decoder(z_q)
                if self.inp_normalizer is not None:
                    z_e_lower = z_e_lower.transpose(1, 2)
                    z_e_lower = self.inp_normalizer.unnorm(z_e_lower)
                    z_e_lower = z_e_lower.transpose(1, 2)
                z_q_lower_tilde = self.prev_model.quantize(z_e_lower)
                recon = self.decode_lower(z_q_lower_tilde)
            else:
                recon = self.decoder(z_q)
        return recon

    def quantize(self, z_e):
        z_q, _ = self.codebook.quantize(z_e)
        return z_q

    def reconstruct(self, x):
        return self.decode(self.quantize(self.encode(x)))

    def recon_loss(self, orig, recon):
        if self.prev_model is None:
            return F.l1_loss(orig, recon)
        else:
            return F.mse_loss(orig, recon)

    def __len__(self):
        i = 1
        layer = self
        while layer.prev_model is not None:
            i += 1
            layer = layer.prev_model
        return i

    def __getitem__(self, idx):
        max_layer = len(self) - 1
        if idx > max_layer:
            raise IndexError("layer does not exist")

        layer = self
        for _ in range(max_layer - idx):
            layer = layer.prev_model
        return layer


class TrainedHQA(TrainedBody):
    """
    Body wrapper class to wrap a trained raw hqa body for benchmarking
    """

    def __init__(self, hqa_model, quantize=True):
        feat_dim = hqa_model.codebook.codebook_groups * hqa_model.codebook.codebook_dim
        super(TrainedHQA, self).__init__(feat_dim=feat_dim, data_class=RawStream)
        self.hqa_model = hqa_model
        self.quantize = quantize

    def forward(self, inputs):
        with torch.no_grad():
            x_mu = mu_law_encoding(inputs).unsqueeze(1)
            x = self.hqa_model.encode(x_mu)
            if self.quantize:
                x, _ = self.hqa_model.codebook.quantize(x)
        return x.permute(0, 2, 1)

    def reconstruct(self, inputs):
        with torch.no_grad():
            x_mu = mu_law_encoding(inputs).unsqueeze(1)
            x_mu_recon = self.hqa_model.reconstruct(x_mu)
            recon_tensor = mu_law_decoding(x_mu_recon)
            recon = recon_tensor.squeeze()
        return recon


class Encoder(nn.Module):
    """ Downsamples by a factor specified by user """

    def __init__(
        self,
        in_feat_dim,
        latent_dim=64,
        downsample=[2, 2],
        hidden_dim=128,
        kernel_size=8,
        groups=4,
        num_layers=4,
    ):
        super().__init__()
        assert all((isinstance(x, int) and x > 0) for x in downsample)
        self.downsample_factor = np.prod(downsample)

        blocks = []

        for i, stride in enumerate(downsample):
            if i == 0:
                blocks.append(
                    EncoderBlock(
                        in_feat_dim,
                        hidden_dim,
                        kernel_size=stride * 2,
                        groups=groups,
                        stride=stride,
                    )
                )
            else:
                blocks.append(
                    EncoderBlock(
                        hidden_dim, hidden_dim, kernel_size=stride * 2, groups=groups, stride=stride
                    )
                )

        blocks.extend(
            [
                EncoderStack(hidden_dim, kernel_size, groups, num_layers),
                Conv1dSamePadding(hidden_dim, latent_dim, kernel_size=1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class EncoderStack(nn.Module):
    def __init__(self, hidden_dim, kernel_size, groups, num_layers=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(EncoderBlock(hidden_dim, hidden_dim, kernel_size, groups))

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1):
        super().__init__()

        blocks = [
            Conv1dSamePadding(in_channels, out_channels, kernel_size, stride=stride),
            nn.GroupNorm(groups, out_channels),
            Mish(),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DecoderMLP(nn.Module):
    """ Upsamples a factor specified by user """

    def __init__(
        self, out_feat_dim, latent_dim=64, hidden_dim=128, upsample=[2, 2],
    ):

        super().__init__()
        assert all((isinstance(x, int) and x > 0) for x in upsample)
        self.receptive_field = 1
        self.max_padding = 1
        self.upsample_factor = np.prod(upsample)

        blocks = []

        for i, scale_factor in enumerate(upsample[::-1]):
            if i == 0:
                blocks.append(
                    DecoderBlock(
                        latent_dim, hidden_dim, kernel_size=scale_factor * 2, upsample=scale_factor
                    )
                )
            else:
                blocks.append(
                    DecoderBlock(
                        hidden_dim, hidden_dim, kernel_size=scale_factor * 2, upsample=scale_factor
                    )
                )

        blocks.extend(
            [
                Permute(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_feat_dim),
                Permute(),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Permute(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Decoder(nn.Module):
    """ Upsamples a factor specified by user """

    def __init__(
        self,
        out_feat_dim,
        latent_dim=64,
        upsample=[2, 2],
        n_residual=128,
        n_skip=128,
        dilation_depth=6,
        n_repeat=4,
        kernel_size=2,
        final_tanh=False,
    ):

        super().__init__()
        assert all((isinstance(x, int) and x > 0) for x in upsample)
        dilations = [kernel_size ** i for i in range(dilation_depth)] * n_repeat
        self.receptive_field = sum(dilations)
        self.max_padding = ceil((dilations[-1] * (kernel_size - 1)) / 2)
        self.upsample_factor = np.prod(upsample)

        blocks = []

        for i, scale_factor in enumerate(upsample[::-1]):
            if i == 0:
                blocks.append(
                    DecoderBlock(
                        latent_dim, n_residual, kernel_size=scale_factor * 2, upsample=scale_factor
                    )
                )
            else:
                blocks.append(
                    DecoderBlock(
                        n_residual, n_residual, kernel_size=scale_factor * 2, upsample=scale_factor
                    )
                )

        blocks.extend(
            [
                ResidualStack(n_residual, n_skip, dilations, kernel_size=kernel_size),
                DecoderBlock(n_skip, (n_skip + out_feat_dim) // 2, kernel_size=1),
                Conv1dSamePadding((n_skip + out_feat_dim) // 2, out_feat_dim, kernel_size=1),
            ]
        )
        if final_tanh is True:
            blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        padding = ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element is True:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class Conv1dMasked(nn.Conv1d):
    def __init__(self, *args, mask_present=False, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

        """ Pad so receptive field sees only frames in past, optionally including present frame """
        if mask_present is True:
            left_padding = ((self.kernel_size[0] - 1) * self.dilation[0]) + 1
        else:
            left_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        self.pad = nn.ConstantPad1d((left_padding, 0), 0)

    def forward(self, x):
        assert x.shape[2] % self.stride[0] == 0
        desired_out_length = x.shape[2] // self.stride[0]
        x = self.pad(x)
        x = super().forward(x)
        return x[:, :, :desired_out_length]


class ResidualStack(nn.Module):
    def __init__(
        self, n_residual, n_skip, dilations, kernel_size=3, groups=4, conv_module=Conv1dSamePadding
    ):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for dilation in dilations:
            self.resblocks.append(
                ResidualBlock(
                    n_residual,
                    n_skip,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    conv_module=conv_module,
                )
            )

    def forward(self, x):
        skip_connections = []
        for resblock in self.resblocks:
            x, skip = resblock(x)
            skip_connections.append(skip)
        return F.relu(sum(skip_connections))


class ResidualBlock(nn.Module):
    def __init__(
        self, n_residual, n_skip, kernel_size=3, dilation=1, groups=4, conv_module=Conv1dSamePadding
    ):
        super().__init__()
        self.conv_tanh = conv_module(
            n_residual, n_residual, kernel_size=kernel_size, dilation=dilation
        )
        self.conv_sigmoid = conv_module(
            n_residual, n_residual, kernel_size=kernel_size, dilation=dilation
        )
        self.gated_activation_unit = GatedActivation(self.conv_tanh, self.conv_sigmoid)

        self.skip_connection = conv_module(n_residual, n_skip, kernel_size=1, dilation=dilation)
        self.residual_connection = conv_module(
            n_residual, n_residual, kernel_size=1, dilation=dilation
        )

    def forward(self, inp):
        x = self.gated_activation_unit(inp)
        skip = self.skip_connection(x)
        residual = self.residual_connection(x)
        output = residual + inp
        return output, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=1):
        super().__init__()

        blocks = [Conv1dSamePadding(in_channels, out_channels, kernel_size), nn.ReLU()]
        if upsample > 1:
            blocks.append(Upsample(scale_factor=upsample))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class GatedActivation(nn.Module):
    def __init__(self, conv_tanh, conv_sigmoid):
        super().__init__()
        self.conv_tanh = conv_tanh
        self.conv_sigmoid = conv_sigmoid
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.tanh(self.conv_tanh(x))
        s = self.sigmoid(self.conv_sigmoid(x))
        return t * s


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class VQCodebook(nn.Module):
    def __init__(self, codebook_slots, codebook_dim, codebook_groups=1, temperature=0.4):
        super().__init__()
        self.codebook_slots = codebook_slots
        self.codebook_dim = codebook_dim
        self.codebook_groups = codebook_groups
        self.latent_dim = codebook_dim * codebook_groups
        self.codebook = nn.Parameter(torch.randn(codebook_slots, codebook_dim))
        self.register_buffer("temperature", torch.tensor(temperature))

    def z_e_to_z_q(self, z_e, soft=True):
        bs, feat_dim, w = z_e.shape
        assert feat_dim == self.latent_dim
        z_e = z_e.view(bs, self.codebook_groups, self.codebook_dim, w)

        z_e = z_e.permute(0, 1, 3, 2).contiguous()
        z_e_flat = z_e.view(bs * self.codebook_groups * w, self.codebook_dim)
        codebook_sqr = torch.sum(self.codebook ** 2, dim=1, keepdim=True)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)

        distances_sqr = torch.addmm(
            codebook_sqr.t() + z_e_flat_sqr, z_e_flat, self.codebook.t(), alpha=-2.0, beta=1.0
        )

        if soft is True:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances_sqr)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, self.codebook_groups, w)
            z_q = (soft_onehot @ self.codebook).view(bs, self.codebook_groups, w, self.codebook_dim)

            # entropy loss
            # TODO: Use dist built in log_probs
            KL = (dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))).mean()

            # probability-weighted commitment loss
            commit_loss = (dist.probs * distances_sqr).mean()
        else:
            with torch.no_grad():
                dist = Categorical(logits=-distances_sqr)
                hard_indices = dist.sample().view(bs, self.codebook_groups, w)
                hard_onehot = (
                    F.one_hot(hard_indices, num_classes=self.codebook_slots)
                    .type_as(self.codebook)
                    .view(bs * self.codebook_groups * w, self.codebook_slots)
                )
                z_q = (hard_onehot @ self.codebook).view(
                    bs, self.codebook_groups, w, self.codebook_dim
                )

                # entropy loss
                KL = (
                    dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))
                ).mean()

                commit_loss = 0.0

        z_q = z_q.permute(0, 1, 3, 2).contiguous()
        z_q = z_q.view(bs, self.latent_dim, w)

        return z_q, hard_indices, KL, commit_loss

    def lookup(self, ids: torch.Tensor):
        bs, code_groups, s = ids.shape
        codes = F.embedding(ids, self.codebook)
        codes = codes.permute(0, 1, 3, 2).contiguous()
        return codes.view(bs, code_groups * self.codebook_dim, s)

    def quantize(self, z_e, soft=False):
        with torch.no_grad():
            z_q, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return z_q, indices

    def quantize_indices(self, z_e, soft=False):
        with torch.no_grad():
            _, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return indices

    def forward(self, z_e):
        z_q, indices, kl, commit_loss = self.z_e_to_z_q(z_e, soft=True)
        return z_q, indices, kl, commit_loss


class HQAGraph(nn.Module):
    """ Trace through encode and (hard) decode to return graph, tracing require forward method """

    def __init__(self, hqa):
        super().__init__()
        self.hqa = hqa

    def reconstruct_hard(self, x):
        hqa = self.hqa
        return hqa.decode(hqa.quantize(hqa.encode(x), soft=False), soft=False)

    def forward(self, x):
        return self.reconstruct_hard(x)
