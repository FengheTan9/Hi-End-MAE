from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def eval_mode(module: nn.Module, enable_dropout: bool = False):
    """Copypasted from pl_bolts.callbacks.ssl_online.set_training
    """
    original_mode = module.training

    try:
        module.eval()
        if enable_dropout:
            for m in module.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        yield module
    finally:
        module.train(original_mode)
