# -*- coding: utf-8 -*-
"""
Methods that are shared across other pieces of code.
Also where all of the flags are defined
"""
from functools import partial
import glob
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
from typing import Tuple

from absl import flags, logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats.mstats import winsorize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import types


flags.DEFINE_integer("seed", 42, "fixed seed to apply to all rng entrypoints")
flags.DEFINE_boolean("debug", False, "enable debugging output")
FLAGS = flags.FLAGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_non_empty_file(path):
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and path.stat().st_size != 0


def parse_audio_dbl(dbl_path):
    dbl_entries = []
    with open(dbl_path) as in_f:
        for line in in_f.readlines():
            line = line.strip()
            if is_non_empty_file(line):
                dbl_entries.append(line)
    if not dbl_entries:
        raise KeyError("dbl list is empty, check paths to dbl files")
    return dbl_entries


def prepare_tb_logging(path=None):
    """
    Ensures that the dir for logging exists and returns a tensorboard logger.
    """
    from torch.utils.tensorboard import SummaryWriter  # dot

    if not path:
        path = FLAGS.expdir
    logdir_path = Path(path)
    logdir_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir_path, flush_secs=10)


def prepare_standard_logging(source):
    """
    Normal Python logging.  None of this TensorBoard new fangled stuff
    """
    # logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)
    logging.set_verbosity(logging.INFO)
    logging.info("Starting {}".format(source))
    logging.info("Flags:")
    logging.info(dict(sorted(FLAGS.flag_values_dict().items())))


def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path):
    try:
        m = torch.load(model_path)
    except RuntimeError:
        m = torch.load(model_path, map_location="cpu")
    return m


class FixedRandomState:
    def __init__(self, seed=0):
        self.seed = seed

    def __enter__(self):
        # Copy current state
        self.random_state = RandomStateCache()

        # Overwrite seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, *args):
        self.random_state.restore()


class RandomStateCache:
    def __init__(self):
        self.store()

    def store(self):
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()

    def restore(self):
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)


class RAdam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020
# github: https://github.com/lessw2020/mish
def mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)


class EmptyScheduler(LambdaLR):
    """ Scheduler that doesn't alter learning rate """

    def __init__(self, optimizer):
        super().__init__(optimizer, lambda x: 1)


# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
class FlatCA(_LRScheduler):
    def __init__(self, optimizer, steps, eta_min=0, last_epoch=-1, decay_proportion=1.0 / 3):
        self.steps = steps
        self.eta_min = eta_min
        self.decay_proportion = decay_proportion
        super(FlatCA, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        T_max = self.steps * self.decay_proportion
        start_step = self.steps - T_max

        for base_lr in self.base_lrs:
            # flat at first
            if 0 <= self._step_count < start_step:
                lr_list.append(base_lr)
            # annealed in last proportion
            else:
                lr_list.append(
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * (self._step_count - start_step) / T_max))
                    / 2
                )
        return lr_list


class BatchNorm(torch.nn.Module):
    """
    nn.Module to handle turning batch norm on or off within the model
    """

    def __init__(self, num_features, batch_norm_on):
        super().__init__()

        self.num_features = num_features
        self.batch_norm_on = batch_norm_on

        if batch_norm_on:
            self.bn = torch.nn.BatchNorm1d(num_features)
        else:
            self.bn = torch.nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


class GlobalNormalization(torch.nn.Module):
    """
    nn.Module to track and normalize input variables, calculates running estimates of data
    statistics during training time.
    Optional scale parameter to fix standard deviation of inputs to 1
    Normalization atlassian page:
    https://speechmatics.atlassian.net/wiki/spaces/INB/pages/905314814/Normalization+Module
    Implementation details:
    "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    """

    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, 1, self.feature_dim))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer("running_sq_diff", torch.zeros(1, 1, self.feature_dim))

    def forward(self, inputs):
        # disabling pylint on a couple of lines as it is bugged at present:
        # TODO: re-enable when pylint is fixed
        # https://github.com/PyCQA/pylint/issues/2315
        # pylint: disable=E0203
        # Check input is of correct shape and matches feature size
        if len(inputs.shape) != 3 or inputs.shape[2] != self.feature_dim:
            raise ValueError(
                f"Inputs do not match required shape [batch_size, window_size, feature_dim], "
                "got {inputs.shape}"
            )
        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * inputs.shape[1]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen + inputs.sum(dim=(0, 1), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)

            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) * (inputs - updated_running_ave)
                ).sum(dim=(0, 1), keepdim=True)

            self.running_ave = updated_running_ave
            self.total_frames_seen = self.total_frames_seen + frames_in_input

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs

    def unnorm(self, inputs):
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs * std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.alpha)
        return loss


def save_checkpoint(path, step, model, optimizer, amp=None, scheduler=None, best_val_loss=None):

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "random_state": RandomStateCache(),
    }

    if amp is not None:
        checkpoint["amp"] = amp.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, amp=None, scheduler=None):
    """ modifying state_dicts in-place """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.restored_step = checkpoint["step"]

    if amp is not None:
        amp.load_state_dict(checkpoint["amp"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

    # TODO: Breaks when checkpoint loaded on different n gpus to that it was saved on.
    # Even if fixed, dl issues mean reloading is still non-determinisitc so dl needs fixing first.
    # checkpoint["random_state"].restore()


def get_checkpoint_to_start_from(checkpoint_path):
    candidate_checkpoints = glob.glob(f"{checkpoint_path}*")
    if candidate_checkpoints:
        candidate_checkpoints.sort(key=os.path.getctime)
        return candidate_checkpoints[-1]
    else:
        return None


def write_current_pid(expdir, pid_path="pid_rank0"):
    os.makedirs(expdir, exist_ok=True)
    with open(os.path.join(expdir, pid_path), "w") as pid:
        pid.write(str(os.getpid()) + "\n")
        pid.flush()


def dump_checkpoint_on_kill(
    model, optimizer, scheduler, checkpoint_out, amp=None, rank=0, ngpus=1, best_val_loss=None
):
    """Handle kill SIGUSR2 signal by checkpointing the very latest model"""

    def signal_handler(signal_num, frame):
        if rank == 0:
            print(f"received kill signal {signal_num}")
            step = scheduler._step_count
            out_path = checkpoint_out + ".step" + str(step) + ".preempted"
            print(f"Saving {out_path}")
            save_checkpoint(out_path, step, model, optimizer, amp, scheduler, best_val_loss)
            os._exit(2)
        if ngpus > 1:
            dist.barrier()

    signal.signal(signal.SIGUSR2, signal_handler)


def extract_flags(FLAGS):
    """ Return abseil flags as a static namespace """
    return types.SimpleNamespace(**{k: v.value for k, v in FLAGS.__flags.items()})


def snapshot_memory_usage():
    if not torch.cuda.is_available():
        return 0, 0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    memory = torch.cuda.memory_allocated() / (1024 ** 3)
    torch.cuda.reset_max_memory_allocated()
    return memory, peak_memory


def wav_to_float(x):
    """
    Input in range -2**15, 2**15 (or what is determined from dtype)
    Output in range -1, 1
    """
    assert x.dtype == torch.int16, f"got {x.dtype}"
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min
    if not x.is_floating_point():
        x = x.to(torch.float)
    x = x - min_value
    x = x / ((max_value - min_value) / 2.0)
    x = x - 1.0
    return x


def float_to_wav(x):
    """
    Input in range -1, 1
    Output in range -2**15, 2**15 (or what is determined from dtype)
    """
    assert x.dtype == torch.float
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min

    x = x + 1.0
    x = x * (max_value - min_value) / 2.0
    x = x + min_value
    x = x.to(torch.int16)
    return x


def mu_law_encoding(x, mu=255.0):
    """
    Input in range -2**15, 2*15 (or what is determined from dtype)
    Output is in range -1, 1 on mu law scale
    """
    x = wav_to_float(x)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * (torch.log1p(mu * torch.abs(x)) / torch.log1p(mu))
    return x_mu


def mu_law_decoding(x_mu, mu=255.0):
    """
    Input is in range -1, 1 on mu law scale
    Output in range -2**15, 2*15 (or what is determined from dtype)
    """
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    x = torch.sign(x_mu) * (1 / mu) * (((1 + mu) ** torch.abs(x_mu)) - 1)
    x = float_to_wav(x)
    return x


def resample_1d(x: torch.Tensor, out_size: int) -> torch.Tensor:
    """
    Used for downsampling labels tensor to match sample rate of predictions tensor
        PyTorch nn.interpolate currently does not support 1d interpolation
        When out_size << x.shape[1] sparse labels may be lost during resample/interpolation
    """
    assert x.ndim == 2, "input should have shape [N, L]"
    in_size = x.shape[1]
    samples = np.linspace(0, in_size - 1, num=out_size)
    resampled_x = interp1d(np.arange(in_size), x, kind="nearest")(samples)
    return torch.tensor(resampled_x, dtype=x.dtype, device=x.device)


def aggregate_time_stats(times: list, device, workers=1) -> Tuple[float, float]:
    """ Calculate statistics for timings, potentially across processes """

    times = torch.tensor(times, device=device)

    mean_time = times.mean()
    max_time = times.max()

    # can replace with dist.reduce, but all_reduce is cheap here and enforces consistency
    if workers > 1:
        dist.all_reduce(mean_time, op=dist.ReduceOp.SUM)
        mean_time /= workers
        dist.all_reduce(max_time, op=dist.ReduceOp.MAX)

    return mean_time.item(), max_time.item()


def setup_dry_run(FLAGS, dry_step=2):
    """ Modify FLAGS to force validation, logging etc after a small number of steps """

    for flag in FLAGS:
        if flag.endswith("_every"):
            setattr(FLAGS, flag, dry_step)
            logging.info(f"setting FLAGS.{flag} = {getattr(FLAGS, flag)}")

    # take additional step to check we can still avoid OOM errors after val, logging etc
    FLAGS.steps = dry_step + 1
    logging.info(f"setting FLAGS.steps = {FLAGS.steps}")

    # create dir to check outputs and avoid polluting expdir
    dry_path = Path(FLAGS.expdir) / "dry_run"

    if dry_path.is_dir():
        shutil.rmtree(dry_path)
    dry_path.mkdir()

    FLAGS.expdir = str(dry_path)
    logging.info(f"setting FLAGS.expdir = {FLAGS.expdir}")


def save_model_with_timestamp(model, output_path):
    """ Save model with timestamp and symlink to output_path """
    output_path_with_timestamp = str(output_path).replace(".pt", f'_{time.strftime("%d%m%y")}.pt')

    torch.save(model, output_path_with_timestamp)

    if os.path.islink(output_path):
        os.unlink(output_path)
    os.symlink(output_path_with_timestamp, output_path)
    print(f"linking: {output_path} -> {output_path_with_timestamp}")


class ModuleHooks:
    """ Register hooks to store activations/grads in a {module: (tensors_in, tensors_out)} dict """

    def __init__(self, modules, hook_type="forward"):
        if hook_type == "forward":
            self.hooks = [m.register_forward_hook(self.hook_fn) for m in modules]
        elif hook_type == "backward":
            self.hooks = [m.register_backward_hook(self.hook_fn) for m in modules]
        self.hooked_values = {}

    def hook_fn(self, module, tensors_in, tensors_out):
        self.hooked_values[module] = (tensors_in, tensors_out)


class TensorHooks:
    """ Register hooks to store grads in a {tensor: grad} dict """

    def __init__(self, tensors):
        self.hooks = [t.register_hook(partial(self.hook_fn, t)) for t in tensors]
        self.hooked_values = {}

    def hook_fn(self, tensor, grad):
        self.hooked_values[tensor] = grad


def log_tb_histogram(tb_logger, values, name, step, clip=0.0):
    """
    clip defines the top/bottom percentile of values which are clipped to constrain outliers
    e.g. clip=0.1 will set all values > 90th percentile to be equal to 90th percentile
    and values < 10th percentile to be equal to 10th percentile
    """
    name = name if clip == 0.0 else f"{name}_w{clip}"
    tb_logger.add_histogram(name, winsorize(values, [clip, clip]), step)
