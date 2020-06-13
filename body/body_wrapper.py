import torch.nn as nn
from typing import Tuple, Optional
from torch import tensor

from abc import ABC, abstractmethod
from data.streaming import FbankStream


class TrainedBody(ABC, nn.Module):
    def __init__(self, feat_dim, data_class):
        """
        Init here is used to add all the attributes that are expected in the model,
        this should be called in every child class __init__ function
        :param feat_dim: feature dimension of outputs
        :param data_class: Expected SingleFileStream class of inputs
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.data_class = data_class

    @property
    def num_params(self) -> int:
        """
        :return: number of parameters in the module
        """
        return sum([p.nelement() for p in self.parameters()])

    @abstractmethod
    def forward(self, inp) -> Tuple[tensor, Optional[tensor]]:
        """
        Extract features for downstream tasks, don't calculate gradients
        :param inp: input into forward pass
            expects type Tensor([batch_size, window_size, input_feat_dim])
        :param state: optional recurrent state
        :return features: features after forward pass through body
        :return state: optional updated recurrent state
        """
        raise NotImplementedError

    def stash_state(self):
        """
        stash state within model and initialize new hidden state
        for use with recurrent models, dummy method for others
        """
        pass

    def pop_state(self):
        """
        pop state from stashed state, overwriting current hidden state
        for use with recurrent models, dummy method for others
        """
        pass

    def reset_state(self):
        """
        reset state for use with recurrent models, dummy method for others
        """
        pass


class EmptyBody(TrainedBody):
    def __init__(self, feat_dim=80, data_class=FbankStream):
        super().__init__(feat_dim, data_class)

    def forward(self, inp):
        return inp
