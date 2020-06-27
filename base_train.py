from absl import flags, logging
import torch

from util import setup_dry_run, get_checkpoint_to_start_from, extract_flags
from distributed import distributed_init

FLAGS = flags.FLAGS
flags.DEFINE_string("model_out", None, "path to where to save trained model")
flags.DEFINE_string("checkpoint", None, "path to loading saved checkpoint")
flags.DEFINE_string("checkpoint_out", None, "path to save checkpoint")
flags.DEFINE_boolean("checkpoint_autoload", True, "if True start from latest checkpoint_out")

flags.DEFINE_string("train_data", None, "path to train files")
flags.DEFINE_string("val_data", None, "path to validation files")
flags.DEFINE_string("expdir", None, "directory to write all experiment data to")

flags.DEFINE_integer("batch_size", None, "batch size, num parallel streams to train on at once")
flags.DEFINE_integer("steps", None, "number of train steps before breaking")
flags.DEFINE_integer("grad_acc_steps", 1, "number batches to accumulate grads before stepping")
flags.DEFINE_float("lr", 4e-4, "learning rate")

flags.DEFINE_integer("dl_max_workers", 8, "maximum number of dataloader subprocesses")
flags.DEFINE_integer("n_gpus", None, "do not set; automatically detected from CUDA_VISIBLE_DEVICES")
flags.DEFINE_string("amp", "O1", "apex amp setting")

flags.DEFINE_integer("val_every", None, "how often to perform validation")
flags.DEFINE_integer("save_every", None, "save every n steps")
flags.DEFINE_integer("log_every", 1, "append to log file every n steps")
flags.DEFINE_integer("log_tb_every", 50, "save tb scalars every n steps")
flags.DEFINE_integer("log_tb_viz_every", 500, "save vizualisations every n steps")

flags.DEFINE_boolean("dry_run", False, "dry run")

flags.mark_flag_as_required("batch_size")
flags.mark_flag_as_required("steps")
flags.mark_flag_as_required("train_data")
flags.mark_flag_as_required("val_data")
flags.mark_flag_as_required("expdir")


def train_init(train, unused_argv):
    """
    N.B. Wrap with partial when passing to app.run as that requires a single callable
    without additional arguments e.g. app.run(partial(train_init, train))
    """

    if FLAGS.dry_run is True:
        setup_dry_run(FLAGS)

    if not FLAGS.model_out:
        FLAGS.model_out = FLAGS.expdir + "/model.pt"
    if not FLAGS.checkpoint_out:
        FLAGS.checkpoint_out = FLAGS.expdir + "/checkpoint.pt"

    # Additional FLAG parsing
    if FLAGS.checkpoint_autoload is True and not FLAGS.checkpoint:
        FLAGS.checkpoint = get_checkpoint_to_start_from(FLAGS.checkpoint_out)
        logging.info(f"autosetting checkpoint: {FLAGS.checkpoint}")

    FLAGS.n_gpus = torch.cuda.device_count()

    # train method to be defined at endpoint
    if FLAGS.n_gpus > 1:
        distributed_init(train, FLAGS.n_gpus, extract_flags(FLAGS))
    else:
        train(FLAGS)
