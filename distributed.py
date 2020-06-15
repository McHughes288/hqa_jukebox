import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def train_worker(rank, world_size, train_fn, *args):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    train_fn(*args, rank=rank)
    dist.destroy_process_group()


def distributed_init(train_fn, n_gpus, *args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["NCCL_DEBUG"] = "INFO"
    mp.spawn(train_worker, args=(n_gpus, train_fn, *args), nprocs=n_gpus, join=True)


def free_port():
    """
    Determines a free port using sockets.
    https://github.com/SeleniumHQ/selenium/blob/master/py/selenium/webdriver/common/utils.py#L31
    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(("0.0.0.0", 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1]
    free_socket.close()
    return port
