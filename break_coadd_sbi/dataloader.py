"""
Create a PyTorch dataset that samples from a simulator to get its data.

We feed
"""

import torch
from torch.utils.data import IterableDataset
from . import catsampler
from . import imagesampler

class SimulatorDataset(IterableDataset):
    def __init__(
        self,
        prior: catsampler.TwoStarCatalogSimulator,
        forward_model: imagesampler.ImageSampler,
        generator: torch.Generator,
        num_workers: int = 1, # workers help organize data b4 it goes gpu
        batch_size: int = 64, # how many images per batch
        epoch_size: int = 64, # how many batches before we recheck validation loss
        cached: bool = False, # whether to cache the data
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.prior = prior
        self.forward_model = forward_model
        self.generator = generator

        # sometimes we want a fixed dataset that doesn't
        # change every time we run through it
        # if so, we store everything all at once
        self.cached=cached
        if cached:
            self._cached = [self._draw_one() for i in range(self.epoch_size)]

    def __len__(self):
        return self.epoch_size

    def _draw_one(self):
        cats = self.prior.sample_catalogs(self.batch_size,generator=self.generator)
        imgs = self.forward_model.sample_images(cats, generator=self.generator)

        return (
            imgs[:,None,:,:], # batch x 1 x height x width, nnets expect some input channels (in this case 1)
            torch.log(cats.source_fluxes), # batch x 2, nnets expect some output channel (in this case 1)
        )

    def __iter__(self):
        if self.cached:
            for b in self._cached:
                yield b
        else:
            for _ in range(self.epoch_size):
                yield self._draw_one()