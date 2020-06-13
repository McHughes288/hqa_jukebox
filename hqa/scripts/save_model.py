#!/usr/bin/env python3.7

import torch
from hqa.model import TrainedHQA
from util import save_model_with_timestamp

if __name__ == "__main__":
    # Hacky script to save new model from state_dict_path to output_path if API changed
    model_path = (
        "/home/johnh/git/hydra/body/research/hqa_audio/models/vary_groups/layer1_groups10.pt"
    )
    model = torch.load(model_path, map_location={"cuda:0": "cpu"}).cpu()

    trained_model = TrainedHQA(model, quantize=True).cpu()
    output_path = "/cantab/dev/inbetweeners/hydra/models/hqa2a/FS2_g10_zq.pt"
    save_model_with_timestamp(trained_model, output_path)
