from functools import partial
from math import inf, log2, ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags, logging
from torch import save
import torch.distributed as dist

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from data.streaming import MultiStreamDataLoader, DblSampler, DblStream, RawStream
from hqa.model import HQA, Encoder, Decoder, VQCodebook, TrainedHQA
from hqa.plot_util import (
    plot_l2_distances,
    plot_histogram,
    reconstruct_for_tensorboard,
)
from hqa.receptive_field import get_receptive_fields, get_downsample

from base_train import train_init
from util import (
    aggregate_time_stats,
    device,
    dump_checkpoint_on_kill,
    FixedRandomState,
    FlatCA,
    load_checkpoint,
    log_tb_histogram,
    ModuleHooks,
    TensorHooks,
    mu_law_encoding,
    parse_audio_dbl,
    prepare_standard_logging,
    prepare_tb_logging,
    RAdam,
    save_checkpoint,
    set_seeds,
    snapshot_memory_usage,
    write_current_pid,
    GlobalNormalization,
)


plt.switch_backend("agg")
torch.backends.cudnn.benchmark = True

FLAGS = flags.FLAGS
flags.DEFINE_string("prev_model", None, "path to previous model to train on top of")

flags.DEFINE_integer("codebook_slots", 256, "number of unique codes to use")
flags.DEFINE_integer("codebook_dim", 64, "dimensionality of each code")
flags.DEFINE_integer("codebook_groups", 2, "the number of groups that are quantized")
flags.DEFINE_list("enc_strides", [2, 2], "downsample strides of the encoder")
flags.DEFINE_integer("enc_hidden_dim", 128, "hidden dimension for enc convs")
flags.DEFINE_integer("enc_num_layers", 4, "number of layer in the encoder residual stack")
flags.DEFINE_integer("enc_kernel_size", 8, "kernel size in encoder")
flags.DEFINE_integer("dec_n_residual", 128, "hidden size of resblock in decoder")
flags.DEFINE_integer("dec_n_skip", 128, "hidden size of skip connections in decoder")
flags.DEFINE_integer("dec_dilation_depth", 6, "number of times to exponentially grow dilation")
flags.DEFINE_integer("dec_n_repeat", 4, "number of times to repeat dilation schedule")

flags.DEFINE_integer("minimum_batch_size", 32, "batch size for accumulating gradients")
flags.DEFINE_integer("window_size", 128, "frames of audio in each datapoint")
flags.DEFINE_float("gs_temp", 0.4, "max temperature for gumbel softmax")
flags.DEFINE_float("temp_min", 0.0001, "min temperature for gumbel softmax")
flags.DEFINE_float("temp_decay_proportion", 1.0, "what proportion to linearly decay for")
flags.DEFINE_boolean("decay_temp", True, "Decay temperature as training progresses")

flags.DEFINE_float("entropy_beta", 5e-5, "Weight of KL entropy term")
flags.DEFINE_float("commit_beta", 1e-2, "Weight of commit")

flags.DEFINE_integer("val_steps", None, "number of steps to take in validation")


def validate(datastream, model, val_steps):
    model.eval()
    losses = []

    # reset random seed for determisitic validation
    with FixedRandomState():
        for step, val_batch in enumerate(datastream):
            data = val_batch["data"].to(device).unsqueeze(1)
            data_mu = mu_law_encoding(data)

            with torch.no_grad():
                recon, orig, _, _, indices, _, _ = model(data_mu)
                loss = model.recon_loss(orig, recon)
                losses.append(loss)
                if step >= val_steps:
                    break
    model.train()
    if model.prev_model is not None:
        model.prev_model.eval()
    loss = torch.mean(torch.stack(losses))
    return loss, indices


def train(FLAGS, rank=0):

    set_seeds(FLAGS.seed)
    if rank == 0:
        write_current_pid(FLAGS.expdir)
        tb_logger = prepare_tb_logging(FLAGS.expdir)

    prepare_standard_logging("training")

    minimum_batch_size = FLAGS.minimum_batch_size
    accumulation_steps = max(1, minimum_batch_size // (FLAGS.batch_size * FLAGS.n_gpus))
    logging.info(f"accumulating {accumulation_steps} times before stepping")

    # model
    if FLAGS.prev_model:
        prev_model = torch.load(FLAGS.prev_model, map_location="cpu")
        input_feat_dim = prev_model.codebook.latent_dim
        inp_normalizer = GlobalNormalization(input_feat_dim, scale=True)
        final_tanh = False
    else:
        prev_model = None
        input_feat_dim = 1
        inp_normalizer = None
        final_tanh = True

    enc_strides = [int(x) for x in FLAGS.enc_strides]
    # TODO check window size is divisible by the product
    encoder = Encoder(
        input_feat_dim,
        latent_dim=FLAGS.codebook_dim * FLAGS.codebook_groups,
        downsample=enc_strides,
        hidden_dim=FLAGS.enc_hidden_dim,
        num_layers=FLAGS.enc_num_layers,
        kernel_size=FLAGS.enc_kernel_size,
    )
    codebook = VQCodebook(
        FLAGS.codebook_slots, FLAGS.codebook_dim, FLAGS.codebook_groups, FLAGS.gs_temp
    )
    decoder = Decoder(
        input_feat_dim,
        latent_dim=FLAGS.codebook_dim * FLAGS.codebook_groups,
        upsample=enc_strides,
        n_residual=FLAGS.dec_n_residual,
        n_skip=FLAGS.dec_n_skip,
        dilation_depth=FLAGS.dec_dilation_depth,
        n_repeat=FLAGS.dec_n_repeat,
        final_tanh=final_tanh,
    )
    model = HQA(prev_model, encoder, codebook, decoder, inp_normalizer).to(device)
    with open(FLAGS.expdir + "/architecture.log", "w") as logfile:
        logfile.write(repr(model))

    downsample_factor = np.prod(enc_strides)
    properties, throughput, sampling_rate = get_model_properties(
        model, FLAGS.window_size, FLAGS.batch_size, downsample_factor
    )
    with open(FLAGS.expdir + "/properties.log", "w") as logfile:
        logging.info(properties)
        logfile.write(properties)

    # tb_logger.add_graph(HQAGraph(model), train_dataset[0][0].unsqueeze(0).to(device))
    model.to(device)
    optimizer = RAdam(model.parameters(), lr=FLAGS.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level=FLAGS.amp)
    scheduler = FlatCA(optimizer, steps=FLAGS.steps, eta_min=1e-6, decay_proportion=0.5)

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
            sampler=DblSampler(train_dbl, loop_data=True),
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
            window_size=144000,  # 3 seconds for validation
            pad_final=True,
        )
        for _ in range(4)
    ]
    val_datastream = MultiStreamDataLoader(val_streams, device=device)

    if not FLAGS.save_every:
        FLAGS.save_every = FLAGS.val_every
    if not FLAGS.val_steps:
        FLAGS.val_steps = max(20, FLAGS.val_every // 100)

    mem, peak_mem = snapshot_memory_usage()
    logging.info(f"pretrain_mem {mem:.5f}GB, pretrain_peak_mem {peak_mem:.5f}GB")

    logging.info("enabling dynamic checkpointing")
    dump_checkpoint_on_kill(
        model, optimizer, scheduler, FLAGS.checkpoint_out, amp, rank, FLAGS.n_gpus
    )

    best_val_loss = inf
    iterations = 0  # number of iterations from datastream
    loader_times = []
    model_times = []
    code_count = torch.zeros(model.codebook.codebook_slots).to(device)

    # setup hooks for logging
    enc_convs = [m for m in model.encoder.modules() if isinstance(m, torch.nn.modules.conv.Conv1d)]
    dec_convs = [m for m in model.decoder.modules() if isinstance(m, torch.nn.modules.conv.Conv1d)]

    enc_conv0_weight = dict(enc_convs[0].named_parameters())["weight"]
    dec_conv0_weight = dict(dec_convs[0].named_parameters())["weight"]

    if model.inp_normalizer is not None:
        fwd_hooks = ModuleHooks(modules=[model.inp_normalizer], hook_type="forward")
    grad_hooks = TensorHooks(tensors=[enc_conv0_weight, dec_conv0_weight])

    while True:
        for train_batch in train_datastream:
            iterations += 1

            data = train_batch["data"].to(device).unsqueeze(1)
            data_mu = mu_law_encoding(data)

            if FLAGS.n_gpus > 1:
                recon, orig, z_q, z_e, indices, KL, commit_loss = dist_model(data_mu)
            else:
                recon, orig, z_q, z_e, indices, KL, commit_loss = model(data_mu)

            # Calculate losses
            recon_loss = model.recon_loss(orig, recon)
            KL_loss = FLAGS.entropy_beta * KL
            current_commit_beta = burn_in_commit_beta_linear(
                step + 1, FLAGS.steps, FLAGS.commit_beta, burn_in_proportion=0.25
            )
            weighted_commit_loss = current_commit_beta * commit_loss
            loss = recon_loss + KL_loss + weighted_commit_loss

            # forward pass memory usage
            fwd_mem, fwd_peak_mem = snapshot_memory_usage()

            # Step
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if iterations % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # anneal temperature
                if FLAGS.decay_temp is True:
                    model.codebook.temperature = decay_temp_exp(
                        step + 1,
                        FLAGS.steps,
                        temp_base=FLAGS.gs_temp,
                        temp_min=FLAGS.temp_min,
                        decay_proportion=FLAGS.temp_decay_proportion,
                    )

            # backward pass memory usage
            bwd_mem, bwd_peak_mem = snapshot_memory_usage()
            # store dataloader times
            loader_times.append(train_datastream.loader_time)
            model_times.append(train_datastream.model_time)

            # log after backwards pass to avoid memory issues
            if step % FLAGS.log_every == 0 and (iterations % accumulation_steps == 0) and rank == 0:

                bits, max_bits = get_bit_usage(model, indices)
                bit_usage_frac = bits / max_bits

                logging.info(
                    (
                        f"rank {rank} step {step}: loss {loss.item():.5f}, "
                        f"bit_usage_frac {bit_usage_frac:.4f}, {bits:.0f}/{max_bits:.0f} bits used."
                    )
                )

                tb_logger.add_scalar("train/loss", loss.item(), step)
                tb_logger.add_scalar("train/recon_loss", recon_loss.item(), step)
                tb_logger.add_scalar("train/KL_loss", KL_loss.item(), step)
                tb_logger.add_scalar("train/commit_loss", commit_loss.item(), step)
                tb_logger.add_scalar(
                    "train/weighted_commit_loss", weighted_commit_loss.item(), step
                )
                tb_logger.add_scalar("train/bit_usage_frac", bit_usage_frac, step)

                with torch.no_grad():
                    # reconstruct all the way, clip to save memory
                    if data_mu.shape[2] > 64000:
                        clipped_data_mu = data_mu[:, :, :64000]
                    else:
                        clipped_data_mu = data_mu
                    orig_mu_recon = model.reconstruct(clipped_data_mu)
                    orig_loss_recon = F.mse_loss(clipped_data_mu, orig_mu_recon)
                    tb_logger.add_scalar("train/orig_loss_recon", orig_loss_recon, step)

            if step % (FLAGS.log_tb_every) == 0 and (iterations % accumulation_steps == 0):
                loader_mean_time, loader_max_time = aggregate_time_stats(
                    loader_times, device, FLAGS.n_gpus
                )
                model_mean_time, model_max_time = aggregate_time_stats(
                    model_times, device, FLAGS.n_gpus
                )
                loader_times = []
                model_times = []

                if rank == 0:
                    current_temperature = model.codebook.temperature.item()
                    current_lr = scheduler.get_lr()[0]
                    throughput_rate = throughput / (loader_mean_time + model_mean_time)

                    loader_time = train_datastream.loader_time
                    model_time = train_datastream.model_time

                    logging.info(
                        (
                            f"KL_loss {KL_loss.item():.5f}, commit_loss {commit_loss.item():.5f}, "
                            f"temp {current_temperature:.3f}, lr {current_lr:.5f}, "
                            f"dl_time {loader_time:.5f}s, m_time {model_time:.5f}s, "
                            f"throughput_rate {throughput_rate:.5f}s/s, "
                            f"f_peak_mem {fwd_peak_mem:.3f}GB, f_mem {fwd_mem:.3f}GB, "
                            f"b_peak_mem {bwd_peak_mem:.3f}GB, b_mem {bwd_mem:.3f}GB"
                        )
                    )
                    tb_logger.add_scalar("train/temp", current_temperature, step)
                    tb_logger.add_scalar("train/lr", current_lr, step)
                    tb_logger.add_scalar("train/commit_beta", current_commit_beta, step)
                    tb_logger.add_scalar("time/dataloader_batch_mean", loader_mean_time, step)
                    tb_logger.add_scalar("time/dataloader_batch_max", loader_max_time, step)
                    tb_logger.add_scalar("time/model_batch_mean", model_mean_time, step)
                    tb_logger.add_scalar("time/model_batch_max", model_max_time, step)
                    tb_logger.add_scalar("time/throughput_rate", throughput_rate, step)

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
                    if val_data_mu.shape[2] > 144000:
                        clipped_val_data_mu = val_data_mu[:, :, :144000]
                    else:
                        clipped_val_data_mu = val_data_mu
                val_orig, val_recon = reconstruct_for_tensorboard(clipped_val_data_mu, model)
                cb_ = model.codebook.codebook

                tb_logger.add_audio(
                    "recon_samples_0", val_recon[0], step, sample_rate=sampling_rate
                )
                tb_logger.add_audio(
                    "recon_samples_1", val_recon[1], step, sample_rate=sampling_rate
                )
                tb_logger.add_audio(
                    "recon_samples_2", val_recon[2], step, sample_rate=sampling_rate
                )
                tb_logger.add_audio(
                    "recon_samples_3", val_recon[3], step, sample_rate=sampling_rate
                )
                tb_logger.add_audio("original_0", val_orig[0], step, sample_rate=sampling_rate)
                tb_logger.add_audio("original_1", val_orig[1], step, sample_rate=sampling_rate)
                tb_logger.add_audio("original_2", val_orig[2], step, sample_rate=sampling_rate)
                tb_logger.add_audio("original_3", val_orig[3], step, sample_rate=sampling_rate)

                tb_logger.add_figure("train/gram_matrix", plot_l2_distances(cb_), step)
                tb_logger.add_figure(
                    "train/code_usage_histogram", plot_histogram(indices, bins=cb_.shape[0]), step
                )
                # code_images = get_usage_imgs_from_indices(model, indices)

                # tb_logger.add_embedding(cb_, label_img=code_images, global_step=step)

                model.train()
                if model.prev_model is not None:
                    model.prev_model.eval()

                ze = model.encode(data_mu)

                zq_soft, _ = model.codebook.quantize(ze, soft=True)
                zq_hard, _ = model.codebook.quantize(ze, soft=False)

                log_tb_histogram(tb_logger, ze.cpu().flatten(), "emb/ze", step)
                log_tb_histogram(tb_logger, zq_soft.cpu().flatten(), "emb/zq_soft", step)
                log_tb_histogram(tb_logger, zq_hard.cpu().flatten(), "emb/zq_hard", step)

                if model.inp_normalizer is not None:
                    input_ze_normalized = fwd_hooks.hooked_values[model.inp_normalizer][1][0].cpu()
                    log_tb_histogram(
                        tb_logger, input_ze_normalized.cpu().flatten(), "emb/input_ze_norm", step
                    )

                enc_conv0_grad = grad_hooks.hooked_values[enc_conv0_weight].cpu()
                dec_conv0_grad = grad_hooks.hooked_values[dec_conv0_weight].cpu()
                log_tb_histogram(
                    tb_logger, enc_conv0_grad.cpu().flatten(), "grad/enc_conv0", step, clip=0.05
                )
                log_tb_histogram(
                    tb_logger, dec_conv0_grad.cpu().flatten(), "grad/dec_conv0", step, clip=0.05
                )

            # Track code usage over time
            with torch.no_grad():
                indices_onehot = F.one_hot(
                    indices, num_classes=model.codebook.codebook_slots
                ).float()
                code_count = code_count + indices_onehot.sum(dim=(0, 1, 2))
                del indices_onehot

            # Reinit trick to keep codebook at high bitfrace usage. Reinit at end of 25th step
            # Only check for reinit if over 100,000 codes seen
            # Don't code reset if near end of training where lr is low.
            if (step % 25 == 0) and ((iterations + 1) % accumulation_steps == 0):
                if step < 3 * (FLAGS.steps // 4):  # and torch.sum(code_count) > 100000:
                    if FLAGS.n_gpus > 1:
                        dist.all_reduce(code_count)
                    with torch.no_grad():
                        max_count, most_used_code = torch.max(code_count, dim=0)
                        frac_usage = code_count / max_count
                        z_q_most_used = model.codebook.lookup(
                            most_used_code.view(1, 1, 1)
                        ).squeeze()

                        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
                        if min_frac_usage < 0.03:
                            logging.info(f"reset code {min_used_code}")
                            with FixedRandomState():
                                moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / 100
                                model.codebook.codebook[min_used_code] = moved_code
                        code_count = torch.zeros_like(code_count)

            # validate periodically
            if step % FLAGS.val_every == 0 and (iterations % accumulation_steps == 0) and rank == 0:
                val_loss, val_indices = validate(val_datastream, model, FLAGS.val_steps)

                bits, max_bits = get_bit_usage(model, val_indices)
                bit_usage_frac = bits / max_bits
                tb_logger.add_scalar("val/bit_usage_frac", bit_usage_frac, step)

                logging.info(f"step {step}, validation loss={val_loss.item():.3}")
                tb_logger.add_scalar("val/loss", val_loss, step)

                if val_loss.item() < best_val_loss:
                    logging.info("saving new best validation")
                    ext = ".bestval"
                    save(model, FLAGS.model_out + ext)
                    save_checkpoint(
                        FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler
                    )
                    best_val_loss = val_loss.item()
                    # save trained hqa with body wrapper
                    trained_model = TrainedHQA(model, quantize=True)
                    save(trained_model, FLAGS.model_out + ext + ".zq.trained")
                    trained_model = TrainedHQA(model, quantize=False)
                    save(trained_model, FLAGS.model_out + ext + ".ze.trained")
            # save out model periodically
            if (
                step % FLAGS.save_every == 0
                and (iterations % accumulation_steps == 0)
                and rank == 0
            ):
                logging.info("Saving out model")
                ext = ".step" + str(step)
                save(model, FLAGS.model_out + ext)
                save_checkpoint(FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler)

            if iterations % accumulation_steps == 0:
                step += 1

            if step > FLAGS.steps:
                break

        if step > FLAGS.steps:
            break

    if rank == 0:
        logging.info("Saving final model")
        save(model, FLAGS.model_out)
        ext = ".step" + str(step)
        save_checkpoint(FLAGS.checkpoint_out + ext, step, model, optimizer, amp, scheduler)
        # save trained hqa with body wrapper
        trained_model = TrainedHQA(model, quantize=True)
        save(trained_model, FLAGS.model_out + ".zq.trained")
        trained_model = TrainedHQA(model, quantize=False)
        save(trained_model, FLAGS.model_out + ".ze.trained")

    # Ensures all processes exit at the same time
    if FLAGS.n_gpus > 1:
        dist.barrier()


def get_model_properties(model, window_size, batch_size, downsample_factor):
    model_properties = []

    # parameter counts
    model_properties.append(
        f"encoder param count {sum(x.numel() for x in model.encoder.parameters()):,}"
    )
    model_properties.append(
        f"decoder param count {sum(x.numel() for x in model.decoder.parameters()):,}"
    )
    model_properties.append(
        f"codebook param count {sum(x.numel() for x in model.codebook.parameters()):,}"
    )
    model_properties.append(f"total param count {sum(x.numel() for x in model.parameters()):,}\n")

    # throughput, downsample and compression
    sampling_rate = RawStream.SAMPLING_RATE_HZ
    throughput = window_size * batch_size / sampling_rate
    model_properties.append(f"throughput {throughput:.2f}s")

    overall_downsample_fac = get_downsample(model, window_size=window_size)
    out_hz = int(sampling_rate // overall_downsample_fac)
    model_properties.append(f"stack downsample factor {int(overall_downsample_fac)} ({out_hz}Hz)")

    pseudo_compression_factor = overall_downsample_fac / model.codebook.codebook_groups
    model_properties.append(f"pseudo compression factor {pseudo_compression_factor:.1f}")

    original_bit_rate = sampling_rate * 16.0 / 1000.0
    bit_rate = (
        log2(model.codebook.codebook_slots) * model.codebook.codebook_groups * out_hz / 1000.0
    )
    model_properties.append(f"bit rate {bit_rate:.1f}kb/s, original {original_bit_rate:.1f}kb/s")

    true_compression_factor = original_bit_rate / bit_rate
    model_properties.append(f"true compression factor {true_compression_factor:.2f}\n")

    # max padding percentage
    latent_window_size = int(window_size // overall_downsample_fac)
    padding_percentage = 100 * model.decoder.max_padding / (latent_window_size * downsample_factor)
    model_properties.append(f"resblock max padding {model.decoder.max_padding}")
    model_properties.append(f"window size {window_size}, latent window size {latent_window_size}")
    model_properties.append(
        f"padding percentage for latent window size {padding_percentage:.2f}%\n"
    )

    # Encoder and decoder local/global receptive fields
    # TODO: This is a hack to get working on gpus w smaller memory
    # enc_lrf, enc_grf, dec_lrf, dec_grf = get_receptive_fields(model, window_size=window_size)
    # enc_ms = ", ".join(f"{1000*r/sampling_rate:.1f}ms" for r in enc_grf)
    # dec_ms = ", ".join(f"{1000*r/sampling_rate:.1f}ms" for r in dec_grf)
    # model_properties.append(f"encoder local receptive field: {', '.join(str(r) for r in enc_lrf)}")
    # model_properties.append(
    #     f"encoder global receptive field: {', '.join(str(r) for r in enc_grf)} ({enc_ms})"
    # )
    # model_properties.append(f"decoder local receptive field: {', '.join(str(r) for r in dec_lrf)}")
    # model_properties.append(
    #     f"decoder global receptive field: {', '.join(str(r) for r in dec_grf)} ({dec_ms})"
    # )
    # resblock_rf = [layer.decoder.receptive_field for layer in model]
    # model_properties.append(
    #     f"decoder resblock receptive field: {', '.join(str(r) for r in resblock_rf)}\n"
    # )

    return "\n".join(model_properties), throughput, sampling_rate


def decay_temp_linear(step, total_steps, temp_base, temp_min=0.0001, decay_proportion=1.0):
    """Linear temp decay"""
    stop_decaying_step = int(ceil(total_steps * decay_proportion))
    step_ = min(step, stop_decaying_step)
    factor = 1.0 - (step_ / stop_decaying_step)
    return torch.tensor(temp_min + (temp_base - temp_min) * factor)


def decay_temp_exp(step, total_steps, temp_base, temp_min=0.0001, decay_proportion=1.0):
    """ Exponential temp decay """
    stop_decaying_step = int(ceil(total_steps * decay_proportion))
    decay_const = (1 / stop_decaying_step) * log2(temp_base / temp_min)
    step_ = min(step, stop_decaying_step)
    temp = temp_base * 2 ** (-decay_const * step_)
    return torch.tensor(temp)


def burn_in_commit_beta_linear(step, total_steps, commit_beta, burn_in_proportion=1.0):
    """Linearly increase the commit beta"""
    stop_decaying_step = int(ceil(total_steps * burn_in_proportion))
    step_ = min(step, stop_decaying_step)
    factor = step_ / stop_decaying_step
    return commit_beta * factor


def get_bit_usage(model, indices):
    """ Calculate bits used by latent space vs max possible """
    num_latents = indices.numel()
    avg_probs = (
        F.one_hot(indices, num_classes=model.codebook.codebook_slots).float().mean(dim=(0, 1, 2))
    )
    bits = (-(avg_probs * torch.log2(avg_probs + 1e-10)).sum()) * num_latents
    max_bits = log2(model.codebook.codebook_slots) * num_latents
    return bits, max_bits


def get_usage_imgs_from_indices(model, indices):
    """
    Get the images used for tensorboard to show how much each code is being used
    """
    code_usage = (
        F.one_hot(indices, num_classes=model.codebook.codebook_slots).float().sum(dim=(0, 1))
    )
    rel_code_usage = code_usage / max(code_usage)
    code_images = rel_code_usage.view(-1, 1).repeat(1, 3)
    code_images[:, 1:] = 0  # Make varying shades of red
    code_images = code_images.view(model.codebook.codebook_slots, 3, 1, 1)
    code_images = code_images.repeat(1, 1, 32, 32).permute(0, 1, 2, 3)
    return code_images


if __name__ == "__main__":
    app.run(partial(train_init, train))
