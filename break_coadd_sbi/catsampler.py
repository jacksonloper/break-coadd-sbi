"""Simple fast simulators creating images and catalogs with exactly two stars in each image,
and only one band of data.
"""

import dataclasses
import math
import torch
from .lanczos import lanczos_interp
from . import catalogs


def draw_trunc_pareto(alpha, H, loc, scale, size, torch_generator):
    """Draw a truncated pareto random variable.

    Args:
        alpha: float, exponent of the power law
        H: float, truncation of the power law
        loc: float, location of the power law
        scale: float, scale of the power law
        size: tuple of ints, size of the output
        torch_generator: torch.Generator, random number generator

    Returns:
        (size) tensor of draws from the truncated pareto distribution

    Specifically, let the "standardized"
    truncated pareto distribution be the distribution with PDF

    f(x) propto x^{-alpha-1} I(1<=x<=H)

    Then we return draws from a shifted and scaled version, i.e.

    X ~ f(x)
    Y = X*scale+loc
    """

    # draw from the standardized truncated pareto
    u = torch.rand(size, generator=torch_generator, device=torch_generator.device)
    x_num = H**alpha + u - u*(H**alpha)
    x_denom = H**alpha
    x = (x_num / x_denom)**(-1/alpha)

    # scale and shift
    return x * scale + loc

@dataclasses.dataclass
class TwoStarCatalogSimulator:
    # typical pixel distance between the stars
    # (is always dithered by ~half-pixel)
    inter_star_distance: float = 10

    # typical center of the stars
    # (also dithered by ~half-pixel)
    center: tuple = (0.0,0.0)

    # brightness distributions for each star
    star_flux_exponent: float = 0.01
    star_flux_truncation: float = 100
    star_flux_loc: float = 3.0
    star_flux_scale: float = 3.0

    def sample_catalogs(self, batch_size, generator) -> catalogs.CatalogBatchFixedStarNum:
        """Sample a CatalogBatch with batch_size images."""

        # get fluxes for both sources
        source_fluxes = draw_trunc_pareto(
            self.star_flux_exponent,
            self.star_flux_truncation,
            loc=self.star_flux_loc,
            scale=self.star_flux_scale,
            size=(batch_size, 2),
            torch_generator=generator,
        )

        # and positions for both sources
        typical_positions = torch.tensor([
            [self.center[0]-self.inter_star_distance / 2, self.center[1]],
            [self.center[0]+self.inter_star_distance / 2, self.center[1]],
        ], device=generator.device)
        dither = torch.randn((batch_size, 2, 2), device=generator.device,generator=generator) * 0.5
        positions = typical_positions + dither

        # done
        return catalogs.CatalogBatchFixedStarNum(positions,source_fluxes)



