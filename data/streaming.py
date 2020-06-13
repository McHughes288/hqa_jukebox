from abc import abstractmethod
from absl import logging
from collections import namedtuple
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterable, Iterator, Tuple, List, Sequence, Dict
import collections
import ffmpeg
import itertools as it
import numpy as np
import random
import signal
import struct
import time
import torch
import torchaudio

AugmentParams = namedtuple("AugmentParams", "type factor")


class BatchTimer:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_time = 0.0
        self.model_time = 0.0

    def __iter__(self):
        t = time.perf_counter()

        for data in self.dataloader:
            self.loader_time = time.perf_counter() - t
            t = time.perf_counter()
            yield data
            self.model_time = time.perf_counter() - t
            t = time.perf_counter()


class DblSampler(collections.abc.Iterable):
    """
    Iterator that samples from a dbl with equal probability
    """

    def __init__(self, dbl, loop_data=True):
        """
        Args:
            dbl: iterable dbl that contains entries to be used by streaming objects
            loop_data: specifies whether dbl should be repeated after exhausting all entries
        """
        self.dbl = dbl
        self.loop_data = loop_data

    def __iter__(self):
        while True:
            dbl_indices = random.sample(range(len(self.dbl)), len(self.dbl))
            for dbl_idx in dbl_indices:
                yield self.dbl[dbl_idx], dbl_idx
            if not self.loop_data:
                return


class DeterministicDblSampler(collections.abc.Iterable):
    """
    Iterator that marches through dbl entries deterministically
    """

    def __init__(self, dbl, loop_data=True):
        """
        Args:
            dbl: iterable dbl that contains entries to be used by streaming objects
            loop_data: specifies whether dbl should be repeated after exhausting all entries
        """
        self.dbl = dbl
        self.loop_data = loop_data

    def __iter__(self):
        while True:
            for dbl_idx, dbl_entry in enumerate(self.dbl):
                yield dbl_entry, dbl_idx
            if not self.loop_data:
                return


class SingleFileStream(collections.abc.Iterable):
    """
    Defines a stream over a single file
    Required methods:
        get_window: returns a window of the required size for patching multiple files together
    """

    OFFSET_MS = 0  # offset of first output frame from start of audio file

    @property
    def sampling_rate_hz(self):
        """ Subclasses must define a class level or instance attribute for output sampling rate """
        if hasattr(self, "_sampling_rate_hz"):
            return self._sampling_rate_hz
        elif hasattr(self, "SAMPLING_RATE_HZ"):
            return self.SAMPLING_RATE_HZ
        else:
            raise NotImplementedError

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, x):
        self._sampling_rate_hz = x

    @abstractmethod
    def get_window(self, window_size):
        raise NotImplementedError


class RawStream(SingleFileStream):
    """SingleFileStream that returns raw int16 audio"""

    SAMPLING_RATE_HZ = 44100
    BIT_DEPTH = 16
    FFMPEG_INPUTARGS = {
        "threads": 1,
    }
    FFMPEG_OUTPUTARGS = {
        "format": "wav",
        "acodec": "pcm_s16le",
        "bits_per_raw_sample": BIT_DEPTH,
        "ac": 1,
        "ar": SAMPLING_RATE_HZ,
        "loglevel": "panic",
        "threads": 1,
    }

    def __init__(self, audio_file, window_size=1, is_wav=False):
        """
        Args:
            audio_file: A filepath or filehandle to the required audiofile
            window_size: Default windowsize when iterating over stream
            is_wav: Specify whether the audiofile is wav format; if not file is auto-converted
        """

        self.audio_handle = self.open_handle(audio_file, is_wav)
        self.window_size = window_size
        self.bytes_per_sample, self.dtype = self.read_wav_header(self.audio_handle)
        self.frame_count = 0

    def open_handle(self, audio_file, is_wav):
        if isinstance(audio_file, str):
            if not Path(audio_file).exists():
                raise FileNotFoundError(f"Audio file path not found: {audio_file}")

            if is_wav is True:
                return open(audio_file, "rb")
            else:
                return self.convert2wav(audio_file)

        elif hasattr(audio_file, "read"):
            return audio_file
        else:
            raise RuntimeError(f"Audio reference type unsupported: {str(type(audio_file))}")

    def read_wav_header(self, audio_handle):
        """
        Read wav header to extract bytes per sample and data type
        """
        fid = audio_handle
        logging.debug("waiting to read header")
        file_size, is_big_endian = wavfile._read_riff_chunk(fid)
        logging.debug("header has been read")
        chunk_id = fid.read(4)
        if chunk_id == b"fmt ":
            fmt_chunk = wavfile._read_fmt_chunk(fid, is_big_endian)
            format_tag, channels, fs = fmt_chunk[1:4]
            assert fmt_chunk[6] == self.BIT_DEPTH, "detected unexpected bit_depth"

        while True:
            chunk_id = fid.read(4)
            if chunk_id == b"data":
                break
            elif chunk_id in (b"fact", b"LIST", b"JUNK", b"Fake"):
                self._skip_unknown_chunk(fid, is_big_endian)
            else:
                raise RuntimeError("wav parsing error")

        if is_big_endian:
            fmt = ">I"
            dtype = ">"
        else:
            fmt = "<I"
            dtype = "<"
        bytes_per_sample = self.BIT_DEPTH // 8
        dtype += "i%d" % bytes_per_sample
        _ = struct.unpack(fmt, fid.read(4))[0]

        return bytes_per_sample, dtype

    @staticmethod
    def _skip_unknown_chunk(fid, is_big_endian):
        if is_big_endian:
            fmt = ">I"
        else:
            fmt = "<I"

        data = fid.read(4)
        if data:
            size = struct.unpack(fmt, data)[0]
            _ = fid.read(size)

    def convert2wav(self, dbl_audio):
        process = (
            ffmpeg.input(str(dbl_audio), **self.FFMPEG_INPUTARGS)
            .output("-", **self.FFMPEG_OUTPUTARGS)
            .global_args("-vn")
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=False)
        )
        audio_handle = process.stdout
        return audio_handle

    def get_window(self, window_size):
        in_bytes = self.audio_handle.read(self.bytes_per_sample * window_size)
        if not in_bytes:
            return
        data = np.frombuffer(in_bytes, self.dtype)

        self.frame_count = self.frame_count + data.shape[0]
        return {
            "data": data,
            "frame_count": np.arange(self.frame_count - data.shape[0], self.frame_count),
        }

    def __iter__(self):
        while True:
            window = self.get_window(self.window_size)
            if window is not None:
                yield window
            else:
                self.audio_handle.close()
                return


class FeatureStream(SingleFileStream):
    """ Single File Stream that returns features such as fbanks or mfccs """

    FRAME_LENGTH_MS = 25.0
    FRAME_SHIFT_MS = 10.0
    SAMPLING_RATE_HZ = int(1000 / FRAME_SHIFT_MS)
    OFFSET_MS = FRAME_LENGTH_MS

    def __init__(self, audio_file, window_size, feature_transform="fbank"):
        """
        Args:
            audio_file: A filepath or filehandle to the required audiofile
            window_size: Default windowsize when iterating over stream
            feature_transform: Feature transform to apply (e.g. fbank, mfcc)
        """
        self.frame_shift_samples = int((self.FRAME_SHIFT_MS / 1000.0) * RawStream.SAMPLING_RATE_HZ)
        self.frame_length_samples = int(
            (self.FRAME_LENGTH_MS / 1000.0) * RawStream.SAMPLING_RATE_HZ
        )
        self.window_size = window_size
        # ensure that the RawStream window_size aligns with FbankStream window_size
        self.audio_stream = RawStream(
            audio_file, window_size=self.frame_shift_samples * window_size
        )

        # buffer the left side of the feature window in order to properly generate fbanks
        self.buffer_size = self.frame_length_samples - self.frame_shift_samples
        self.buffer = self.audio_stream.get_window(self.buffer_size)["data"]
        self.frame_count = 0

        if feature_transform == "fbank":
            self.feature_fn = partial(
                torchaudio.compliance.kaldi.fbank,
                window_type="hamming",
                dither=1.0,
                num_mel_bins=80,
                htk_compat=True,
                use_energy=False,
                frame_length=self.FRAME_LENGTH_MS,
                frame_shift=self.FRAME_SHIFT_MS,
            )
        elif feature_transform == "mfcc":
            self.feature_fn = partial(
                torchaudio.compliance.kaldi.mfcc,
                num_mel_bins=40,
                num_ceps=40,
                use_energy=False,
                high_freq=-400,
                frame_length=self.FRAME_LENGTH_MS,
                frame_shift=self.FRAME_SHIFT_MS,
            )
        else:
            raise (f"Feature transform {feature_transform} not supported")

    def get_window(self, window_size):
        """ Applies fbank function over given window, updates buffer """
        raw_window = self.audio_stream.get_window(window_size * self.frame_shift_samples)
        if raw_window is None:
            return
        buffered_window = np.concatenate([self.buffer, raw_window["data"]])
        self.buffer = buffered_window[-self.buffer_size :]
        # check we have sufficient window to obtain at least one fbank
        if buffered_window.shape[0] < self.frame_length_samples:
            return
        data = self.feature_fn(torch.Tensor(buffered_window).unsqueeze(0)).numpy()
        self.frame_count = self.frame_count + data.shape[0]
        return {
            "data": data,
            "frame_idx": np.arange(self.frame_count - data.shape[0], self.frame_count),
        }

    def __iter__(self):
        while True:
            window = self.get_window(self.window_size)
            if window is None:
                return
            yield window


class FbankStream(FeatureStream):
    """ Single File Stream that returns fbanks """

    def __init__(self, audio_file, window_size):
        """
        Args:
            audio_file: A filepath or filehandle to the required audiofile
            window_size: Default windowsize when iterating over stream
        """
        super(FbankStream, self).__init__(audio_file, window_size, feature_transform="fbank")


class MFCCStream(FeatureStream):
    """ Single File Stream that returns mfccs with speechmatics params """

    def __init__(self, audio_file, window_size):
        """
        Args:
            audio_file: A filepath or filehandle to the required audiofile
            window_size: Default windowsize when iterating over stream
        """
        super(MFCCStream, self).__init__(audio_file, window_size, feature_transform="mfcc")


class DblStream:
    """
    Combines a DBL Sampler and Single File Stream to produce an stream with consistent window size
    """

    def __init__(
        self,
        sampler: DblSampler,
        single_file_stream_class: SingleFileStream,
        window_size: int,
        pad_final=False,
        **kwargs,
    ):
        """
        Args:
            sampler: Iterable that returns a dbl entry and dbl index
            single_file_stream_class: any subclass of SingleFileStream that is used to
                iterate over entries in the dbl
            window_size: desired default window size when iterating
        """
        self.sampler = sampler
        self.single_file_stream = single_file_stream_class
        self.window_size = window_size
        self.pad_final = pad_final
        self.kwargs = kwargs

    def __iter__(self):
        frames_left_in_window = 0
        partial_window = None  # Keep track of any samples that don't fill an entire window
        for dbl_entry, dbl_idx in self.sampler:
            stream = self.single_file_stream(dbl_entry, self.window_size, **self.kwargs)
            dbl_idx_repeat = np.repeat(dbl_idx, repeats=self.window_size)
            # If we need to fill up window, grab the required amount from the stream
            if frames_left_in_window > 0:
                window = stream.get_window(frames_left_in_window)
                window["dbl_idx"] = dbl_idx_repeat[: window["data"].shape[0]]
                partial_window = aggregate_dicts([partial_window, window], np.concatenate)
                frames_left_in_window = self.window_size - partial_window["data"].shape[0]
                # Once filled up, yield and move on
                if frames_left_in_window == 0:
                    yield partial_window
                    partial_window = None
            # Loop through stream
            for window in stream:
                frames_left_in_window = self.window_size - window["data"].shape[0]
                window["dbl_idx"] = dbl_idx_repeat[: window["data"].shape[0]]
                # If window_size not left in stream, break to fill up with next file
                if frames_left_in_window > 0:
                    partial_window = window
                    break
                yield window
        # If we reach the end of the sampler and we have a partial window left we return it
        if partial_window is not None:
            if self.pad_final is True:
                padding_window = {
                    key: np.zeros((frames_left_in_window, *array.shape[1:]), dtype=array.dtype)
                    for key, array in partial_window.items()
                }

                padding_window["dbl_idx"] = np.repeat(dbl_idx, repeats=frames_left_in_window)
                # set padded frame indices to -1 for easy filtering
                if "frame_idx" in padding_window.keys():
                    padding_window["frame_idx"] = np.repeat(-1, repeats=frames_left_in_window)

                partial_window = aggregate_dicts([partial_window, padding_window], np.concatenate)

            yield partial_window


class MultiStreamDataset(IterableDataset):
    """
    PyTorch Dataset to combine batches across multiple streams and combine into a single array
    """

    def __init__(self, streams: List[Iterable[Tuple[np.array]]]):
        self.streams = streams

    def __iter__(self) -> Iterator[Tuple[np.array]]:
        for batch in zip(*self.streams):
            yield aggregate_dicts(batch, np.array)


class MultiStreamDataLoader:
    """
    Wrap groups of streams in their own dataloaders to make use of
    multithreading in an efficient way.

    number of streams must be divisible by max_workers to utilise all workers
    """

    def __init__(
        self, streams: List[Iterable[Dict]], max_workers=8, device="cpu", pin_memory=None,
    ):
        self.streams = streams
        self.max_workers = max_workers
        self.device = device
        self.batch_size = len(streams)
        self.split_size = self.maximize_split_size(self.max_workers, self.batch_size)
        self.batch_timer = None

        if not streams:
            raise RuntimeError("streams cannot be empty")

        self.datasets = []
        for i in range(self.batch_size // self.split_size):
            streams = self.streams[i * self.split_size : (i + 1) * self.split_size]
            self.datasets.append(MultiStreamDataset(streams))

        self.dl_workers = min(max_workers, 1)

        if pin_memory is None:
            self.pin_memory = self.device == "cuda"
        else:
            self.pin_memory = pin_memory

    def __iter__(self) -> Iterator[Tuple[torch.Tensor]]:

        stream_loaders = BatchTimer(
            zip(
                *[
                    DataLoader(
                        dataset,
                        num_workers=self.dl_workers,
                        batch_size=None,
                        pin_memory=self.pin_memory,
                        worker_init_fn=worker_init_fn,
                    )
                    for dataset in self.datasets
                ]
            )
        )
        self.batch_timer = stream_loaders
        for sub_batches in stream_loaders:
            output = aggregate_dicts(sub_batches, torch.cat)
            yield output

    @property
    def loader_time(self):
        return self.batch_timer.loader_time

    @property
    def model_time(self):
        return self.batch_timer.model_time

    @staticmethod
    def maximize_split_size(max_workers, batch_size):

        if max_workers == 0:
            return batch_size

        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break

        return batch_size // num_workers


def worker_init_fn(_):
    """
    To ensure that we get different (but deterministic) seeds for each multiprocessing worker
    we map the pytorch random state to a numpy seed.
    As the state contains more information than there are possible seeds we use the first 32 bits.
    Ignoring zeros to get closer to uniform seed distribution as ~50% of the random state == 0.
    """
    state32 = list(it.islice((x.item() for x in torch.random.get_rng_state() if x != 0), 4))
    seed = int.from_bytes(bytes(state32), byteorder="big")
    np.random.seed(seed)

    # worker subprocesses shouldn't handle SIGUSR2; only the parent process should
    # attempt dynamic checkpointing upon this signal
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)


def whole_file_fn(stream: SingleFileStream) -> torch.Tensor:
    """Returns the entire output of a single file stream in one tensor"""
    return aggregate_dicts(list(stream), lambda x: torch.Tensor(np.concatenate(x)))["data"]


def aggregate_dicts(dicts: Sequence[dict], agg_func) -> dict:
    return {key: agg_func([d[key] for d in dicts]) for key in dicts[0]}
