"""Simple fast image simulators catalogs with fixed number of stars in each image, and
only one band of data.
"""

import dataclasses
import math
import torch
from .lanczos import lanczos_interp
from .badcolumns import make_bad_column
from .cosmicray import generate_cosmic_ray

def place_gaussian_bumps(plocs,fluxes,xx,yy,psf_h,psf_w):
    """Place gaussian bumps down on an image grid with coordinates defined
    by xx,yy.

    Args:
        plocs: (B,max_sources,2) tensor of locations
        fluxes: (B,max_sources) tensor of fluxes
        xx: (rows) tensor of x coordinates
        yy: (columns) tensor of y coordinates
        psf_h: float, height of the PSF
        psf_w: float, width of the PSF

    Returns:
        (B,max_sources,height,width) tensor of images

    In the output, the pixel at img[b,i,j,k] spans a box from
    xx[j],yy[k] to xx[j+1],yy[k+1]
    """

    xx=xx+.5
    yy=yy+.5


    # for each b,i,x, we compute the pdf at x when
    # X ~ N(plocs_x[b,i],psf_radius_h^2)
    pdfx = torch.exp(-0.5 * ((xx[None,None] - plocs[:,:,[0]]) / psf_h) ** 2) / (psf_h * math.sqrt(2 * math.pi))

    # same for Y
    pdfy = torch.exp(-0.5 * ((yy[None,None] - plocs[:,:,[1]]) / psf_w) ** 2) / (psf_w * math.sqrt(2 * math.pi))

    # mult it up
    return (
        pdfx[:,:,:,None] *
        pdfy[:,:,None,:] *
        fluxes[:,:,None,None]
    ).sum(1)

@dataclasses.dataclass
class ImageSampler:
    """Sampler for images of stars.

    We assume the following simplified CCD model.  For each single exposure...

    1. We are given a list of stars (a catalog), each with some amount of flux,
       delivered to us in some unspecified (energy / area / time) units.  The positions of the
       stars are assumed to already be in pixel coordinates.
    2. The flux is dispersed via a PSF, which is the same for all bands.  The PSF
       models the effects of the atmosphere/telescope and also the CCD sensitivity.
       The PSF is the same for all bands and all pixels.  After applying the PSF, we get a kind of
       flux value y, in units of energy / pixel / picture.
    3. At each pixel, we assume observations (measured in ADUs) are drawn from
           N(a* y+eps, b*y + c)
       where (a,b,c,eps) together account for dark current, read noise, gain, quantum
       efficiency, background sky flux, and perhaps other things too.

    A final coadd is produced by combining `coadd_depth` single-exposure images.

    Remarks
    * Single-exposure images comprising the coadds are simulated as having been taken at slightly different
      locations, which are corrected for in the final coadd.  This is simulated by first sampling each single
      exposures at a slightly offset position and then correcting for the shift with Lanczos.  If
      `translation_artifacts` is False, then we do not apply this part of the simulations.
    * The PSF is already "pixel-convolved."  That is, the flux at a pixel is calculated by
      evaluating of f at a point, rather than integrating f over a region.  Note that
      there's no reason to suppose the PSF would integrate to 1 or anything like that.  In this sense,
      the PSF could perhaps be considered to account for some of the quantum efficiency properties.
      For the sake of a simple model, we here take the PSF to be a 2d gaussian density with diagonal,
      covariance, diag([psf_radius_h^2,psf_radius_w^2]).
    * Coordinates are such that pixel [0,0] is assumed to span from 0,0 to 1,1, so is centered
      at (.5,.5).  So, e.g., use plt.pcolormesh instead of plt.imshow for rendering images made.
    * We assume every single exposure has the same PSF, and the same value for (a,eps,b,c).
    * There's only one band of imaging.
    """

    # properties of each exposure
    height: int = 80
    width: int = 80

    # psf
    psf_radius_h: float = 2
    psf_radius_w: float = 2

    # noise properties
    # ADU ~ N(a*flux + eps, b*flux + c)
    ccd_a: float = 1.0
    ccd_b: float = 1.0
    ccd_c: float = 1.0
    ccd_eps: float = 1.0

    # whether to include translation artifacts
    translation_artifacts: bool = True

    # whether to include bad column artifacts
    bad_column_artifacts: bool = False

    # whether to include cosmic ray artifacts
    cosmic_ray_artifacts: bool = False

    # coadd depth
    coadd_depth: int = 5

    def __post_init__(self):
        # check a types to make sure they are integer
        if not isinstance(self.height, int):
            raise ValueError("height must be an int")
        if not isinstance(self.width, int):
            raise ValueError("width must be an int")
        if not isinstance(self.coadd_depth, int):
            raise ValueError("coadd_depth must be an int")

    def sample_images(self, catalog, generator) -> torch.tensor:
        """Create images from a catalog and misalignments.

        Args:
            catalog: CatalogBatchFixedStarNum, the catalog of sources
            generator: torch.Generator, random number generator used for making noise

        Returns:
            (B, height, width) tensor, the coadded images

        """

        n_batch = catalog.source_fluxes.shape[0]

        # construct meshgrid of size (height,width)
        # indicating centers of pixels
        xx = torch.arange(self.height, device=generator.device)+.5
        yy = torch.arange(self.width, device=generator.device)+.5

        img = 0
        for coadd in range(self.coadd_depth):

            # translation artifacts: optionally create an offset in the range [-.5,.5]^2
            # for each image
            if self.translation_artifacts:
                offsets = torch.rand((n_batch,2),
                                generator=generator,device=generator.device) - 0.5
            else:
                offsets = torch.zeros(
                    (n_batch,2),
                    device=generator.device
                )

            # get locations (maybe offset)
            locs = catalog.source_locations + offsets[:, None,:]

            # calculate fluxes (in units of energy/pixel/exposure)
            brightness = place_gaussian_bumps(locs,catalog.source_fluxes,
                                    xx,yy,self.psf_radius_h,self.psf_radius_w)

            # get final ADU measurement, with gaussian noise
            rendering = torch.normal(
                brightness * self.ccd_a + self.ccd_eps,
                torch.sqrt(brightness * self.ccd_b + self.ccd_c),
                generator=generator,
            )

            # mask bad column if option is true
            if self.bad_column_artifacts:
                rendering = make_bad_column(rendering, generator)

            # mask cosmic rays if option is true
            if self.cosmic_ray_artifacts:
                rendering = generate_cosmic_ray(rendering, generator)

            # if the CCD was offset, then we have to
            # shift it back into the right coordinates
            if self.translation_artifacts:
                rendering = lanczos_interp(rendering, -offsets)

            # add it into the mix with everything else
            img += rendering

        return img