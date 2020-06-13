from functools import partial
from math import inf

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from absl import app, flags, logging
from torch import save
import torch.distributed as dist

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from data.streaming import MultiStreamDataLoader, DblSampler, DblStream, RawStream
from lsm.model import LSMGRU, TrainedLSM
from base_train import train_init
from util import (
    FixedRandomState,
    RAdam,
    prepare_standard_logging,
    prepare_tb_logging,
    set_seeds,
    device,
    FlatCA,
    save_checkpoint,
    parse_audio_dbl,
    mu_law_encoding,
    mu_law_decoding,
    wav_to_float,
    load_checkpoint,
    write_current_pid,
    snapshot_memory_usage,
)


plt.switch_backend("agg")
torch.backends.cudnn.benchmark = True

FLAGS = flags.FLAGS
flags.DEFINE_string("hqa_path", None, "path to hqa model to use")
flags.DEFINE_integer("hidden_dim", 1024, "Hidden dim of lsm model")
flags.DEFINE_integer("num_layers", 3, "Num layers of lsm model")
flags.DEFINE_integer("minimum_batch_size", 32, "batch size for accumulating gradients")
flags.DEFINE_integer("window_size", 128, "frames of audio in each datapoint")
flags.DEFINE_integer("val_steps", None, "number of steps to take in validation")
flags.mark_flag_as_required("hqa_path")


def validate(datastream, model, val_steps):
    model.eval()
    losses = []

    # reset random seed for determisitic validation
    with FixedRandomState():
        for step, val_batch in enumerate(datastream):
            data = val_batch["data"].to(device).unsqueeze(1)
            data_mu = mu_law_encoding(data)

            with torch.no_grad():
                z_e = model.hqa.encode(data_mu)

                # Ensure the offset is correct so always predict one frame in the future
                z_e_inp = z_e[:, :, :-1]
                z_e_target = z_e[:, :, 1:]

                z_e_pred = model(z_e_inp)

                loss = F.mse_loss(z_e_pred, z_e_target)
                losses.append(loss.item())

                if step >= val_steps:
                    break
    model.train()
    loss = torch.Tensor(losses).mean()
    return loss


def train(FLAGS, rank=0):
    if not FLAGS.model_out:
        FLAGS.model_out = FLAGS.expdir + "/model.pt"
        FLAGS.checkpoint_out = FLAGS.expdir + "/checkpoint.pt"

    set_seeds(FLAGS.seed)
    if rank == 0:
        write_current_pid(FLAGS.expdir)
        tb_logger = prepare_tb_logging(FLAGS.expdir)

    prepare_standard_logging("training")

    minimum_batch_size = FLAGS.minimum_batch_size
    accumulation_steps = max(1, minimum_batch_size // (FLAGS.batch_size * FLAGS.n_gpus))
    logging.info(f"accumulating {accumulation_steps} times before stepping")

    hqa = torch.load(FLAGS.hqa_path, map_location="cpu").to(device)
    hqa.eval()

    model = LSMGRU(
        hqa,
        inp_dim=hqa.codebook.latent_dim,
        out_dim=hqa.codebook.latent_dim,
        hidden_dim=FLAGS.hidden_dim,
        num_layers=FLAGS.num_layers,
    ).to(device)
    """
    model = LSMTransformer(
        inp_dim=hqa.codebook.latent_dim,
        out_dim=hqa.codebook.latent_dim,
        num_heads=4,
        hidden_dim=FLAGS.hidden_dim,
        num_layers=FLAGS.num_layers,
    ).to(device)
    """

    print(f"LSM model param count {sum(x.numel() for x in model.parameters()):,}")
    optimizer = RAdam(model.parameters(), lr=FLAGS.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level=FLAGS.amp)
    scheduler = FlatCA(optimizer, steps=FLAGS.steps, eta_min=1e-6, decay_proportion=1.0)

    step = 0

    if FLAGS.checkpoint:
        # loading state_dicts in-place
        load_checkpoint(FLAGS.checkpoint, model, optimizer, amp=amp, scheduler=scheduler)
        step = optimizer.restored_step

    if FLAGS.n_gpus > 1:
        dist_model = DDP(model)
        model = dist_model.module

    # data
    set_seeds(FLAGS.seed + rank + step)  # Make sure data is different across each process
    train_dbl = parse_audio_dbl(FLAGS.train_data)
    train_streams = [
        DblStream(
            sampler=DblSampler(train_dbl, loop_data=False),
            single_file_stream_class=RawStream,
            window_size=FLAGS.window_size,
        )
        for _ in range(FLAGS.batch_size)
    ]
    train_datastream = MultiStreamDataLoader(train_streams, device=device)

    # setup testing dataloader
    val_dbl = parse_audio_dbl(FLAGS.val_data)
    val_streams = [
        DblStream(
            sampler=DblSampler(val_dbl, loop_data=False),
            single_file_stream_class=RawStream,
            window_size=RawStream.SAMPLING_RATE_HZ * 3,
        )
        for _ in range(4)
    ]
    val_datastream = MultiStreamDataLoader(val_streams, device=device)

    throughput = FLAGS.window_size * FLAGS.batch_size / RawStream.SAMPLING_RATE_HZ

    if not FLAGS.save_every:
        FLAGS.save_every = FLAGS.val_every
    if not FLAGS.val_steps:
        FLAGS.val_steps = max(20, FLAGS.val_every // 100)

    mem, peak_mem = snapshot_memory_usage()
    logging.info(f"pretrain_mem {mem:.5f}GB, pretrain_peak_mem {peak_mem:.5f}GB")

    best_val_loss = inf
    iterations = 0  # number of iterations from datastream
    while True:
        for train_batch in train_datastream:
            iterations += 1

            data = train_batch["data"].to(device).unsqueeze(1)
            data_mu = mu_law_encoding(data)

            z_e = model.hqa.encode(data_mu)

            # Ensure the offset is correct so always predict one frame in the future
            z_e_inp = z_e[:, :, :-1]
            z_e_target = z_e[:, :, 1:]

            if FLAGS.n_gpus > 1:
                z_e_pred = dist_model(z_e_inp)
            else:
                z_e_pred = model(z_e_inp)

            loss = F.mse_loss(z_e_target, z_e_pred)

            # forward pass memory usage
            fwd_mem, fwd_peak_mem = snapshot_memory_usage()

            # Step
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if iterations % accumulation_steps == 0:
                step += 1
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # backward pass memory usage
            bwd_mem, bwd_peak_mem = snapshot_memory_usage()

            loader_time = train_datastream.loader_time
            model_time = train_datastream.model_time
            throughput_rate = throughput / (loader_time + model_time)
            total_throughput_rate = throughput_rate * FLAGS.n_gpus
            logging.info(
                (
                    f"rank {rank} step {step}: loss {loss.item():.5f}. "
                    f"dl_time {loader_time:.5f}s, m_time {model_time:.5f}s, "
                    f"throughput_rate {throughput_rate:.5f}s/s, "
                    f"f_peak_mem {fwd_peak_mem:.3f}GB, f_mem {fwd_mem:.3f}GB, "
                    f"b_peak_mem {bwd_peak_mem:.3f}GB, b_mem {bwd_mem:.3f}GB"
                )
            )

            # log after backwards pass to avoid memory issues
            if step % FLAGS.log_every == 0 and (iterations % accumulation_steps == 0) and rank == 0:
                tb_logger.add_scalar("train/loss", loss.item(), step)
                tb_logger.add_scalar("train/lr", scheduler.get_lr()[0], step)
                tb_logger.add_scalar("time/dataloader_batch", loader_time, step)
                tb_logger.add_scalar("time/model_batch", model_time, step)
                tb_logger.add_scalar("time/throughput_rate", total_throughput_rate, step)

            # Log original and reconstructions
            if (
                step % FLAGS.log_tb_viz_every == 0
                and (iterations % accumulation_steps == 0)
                and rank == 0
            ):
                model.eval()
                with FixedRandomState(1):
                    val_data = next(iter(val_datastream))["data"].to(device)
                    val_data = val_data.unsqueeze(1)
                    val_data_mu = mu_law_encoding(val_data)
                val_orig, val_recon_hqa, val_recon_lsm = reconstruct_for_tensorboard(
                    val_data_mu, model
                )

                tb_logger.add_audio(
                    "original_0", val_orig[0], step, sample_rate=RawStream.SAMPLING_RATE_HZ
                )
                tb_logger.add_audio(
                    "original_1", val_orig[1], step, sample_rate=RawStream.SAMPLING_RATE_HZ
                )
                tb_logger.add_audio(
                    "original_2", val_orig[2], step, sample_rate=RawStream.SAMPLING_RATE_HZ
                )
                tb_logger.add_audio(
                    "hqa_recon_samples_0",
                    val_recon_hqa[0],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )
                tb_logger.add_audio(
                    "hqa_recon_samples_1",
                    val_recon_hqa[1],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )
                tb_logger.add_audio(
                    "hqa_recon_samples_2",
                    val_recon_hqa[2],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )
                tb_logger.add_audio(
                    "lsm_recon_samples_0",
                    val_recon_lsm[0],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )
                tb_logger.add_audio(
                    "lsm_recon_samples_1",
                    val_recon_lsm[1],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )
                tb_logger.add_audio(
                    "lsm_recon_samples_2",
                    val_recon_lsm[2],
                    step,
                    sample_rate=RawStream.SAMPLING_RATE_HZ,
                )

                model.train()

            # validate periodically
            if (step + 1) % (FLAGS.val_every + 1) == 0 and (
                iterations % accumulation_steps == 0 and rank == 0
            ):
                val_loss = validate(val_datastream, model, FLAGS.val_steps)

                logging.info(f"{step} validation, loss={val_loss.item():.3}")
                tb_logger.add_scalar("val/loss", val_loss, step)

                if val_loss.item() < best_val_loss:
                    logging.info("Saving new best validation")
                    ext = ".bestval"
                    save(model, FLAGS.model_out + ext)
                    save_checkpoint(
                        FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler
                    )
                    best_val_loss = val_loss.item()
                    trained_model = TrainedLSM(model)
                    save(trained_model, FLAGS.model_out + ext + ".trained")

            # save out model periodically
            if (
                (step + 1) % (FLAGS.save_every + 1) == 0
                and (iterations % accumulation_steps == 0)
                and rank == 0
            ):
                ext = ".step" + str(step)
                save(model, FLAGS.model_out + ext)
                save_checkpoint(FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler)
            if step > FLAGS.steps:
                break

        if step > FLAGS.steps:
            break

    if rank == 0:
        save(model, FLAGS.model_out)
        ext = ".step" + str(step)
        save_checkpoint(FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler)
        trained_model = TrainedLSM(model)
        save(trained_model, FLAGS.model_out + ext + ".trained")

    # Ensure all processes exit at the same time
    if FLAGS.n_gpus > 1:
        dist.barrier()


def reconstruct_for_tensorboard(x_mu, model):
    """
    Reconstruct all the way to floats in range -1, 1
    """
    model.eval()
    # TODO: Udpate if scheme is changed
    z_e = model.hqa.encode(x_mu)
    z_e_inp = z_e[:, :, :-1]
    with torch.no_grad():
        z_e_pred = model(z_e_inp)
        z_q_pred = model.hqa.quantize(z_e_pred)
    hqa_recon_mu = model.hqa.decode(model.hqa.quantize(z_e))
    lsm_recon_mu = model.hqa.decode(z_q_pred)

    x_mu = x_mu.squeeze(1)
    hqa_recon_mu = hqa_recon_mu.squeeze(1)
    lsm_recon_mu = lsm_recon_mu.squeeze(1)

    x_float = wav_to_float(mu_law_decoding(x_mu)).cpu().numpy()
    hqa_recon_float = wav_to_float(mu_law_decoding(hqa_recon_mu)).cpu().numpy()
    lsm_recon_float = wav_to_float(mu_law_decoding(lsm_recon_mu)).cpu().numpy()

    model.train()
    model.hqa.eval()

    return x_float, hqa_recon_float, lsm_recon_float


if __name__ == "__main__":
    app.run(partial(train_init, train))
