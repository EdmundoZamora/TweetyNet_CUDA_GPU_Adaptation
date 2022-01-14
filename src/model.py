import numpy as np
import torch
import vak

from .network import TweetyNet


def acc(y_pred, y):
    return y_pred.eq(y.view_as(y_pred)).sum().item() / np.prod(y.shape)
"""
This is the model described in the TweetyNet Paper: https://www.biorxiv.org/content/10.1101/2020.08.28.272088v1.full
We do not use it because VAK does not work properly
"""


class TweetyNetModel(vak.Model):
    @classmethod
    def from_config(cls, config, logger=None):
        network = TweetyNet(**config['network'])
        loss = torch.nn.CrossEntropyLoss(**config['loss'])
        optimizer = torch.optim.Adam(params=network.parameters(), **config['optimizer'])
        metrics = {'acc': vak.metrics.Accuracy(),
                   'levenshtein': vak.metrics.Levenshtein(),
                   'segment_error_rate': vak.metrics.SegmentErrorRate(),
                   'loss': torch.nn.CrossEntropyLoss()}
        return cls(network=network, optimizer=optimizer, loss=loss, metrics=metrics, logger=logger)
