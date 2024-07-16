import dataclasses
import torch

@dataclasses.dataclass
class CatalogBatchFixedStarNum:
    """Catalogs for a batch of B images

    In this toy model, every image contains exactly N star sources, and there's only
    one band of imaging.
    """

    source_locations: torch.Tensor  #  (N,2,2), locations of the sources
    source_fluxes: torch.Tensor     #  (N,2), measure of flux for each star each band

    def __post_init__(self):
        """Check that the input tensors have the right shape, and set some
        useful attributes."""

        if len(self.source_fluxes.shape) != 2:
            raise ValueError("source_fluxes should have shape (N,2)")

        self.n_batch = self.source_fluxes.shape[0]

        if self.source_fluxes.shape != (self.n_batch, 2):
            raise ValueError("source_fluxes should have shape (N,2)")
        if self.source_locations.shape != (self.n_batch, 2, 2):
            raise ValueError("source_locations should have shape (N,2,2)")
