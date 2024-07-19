"""Simple method to simulate bad columns to a generated image.
"""

import torch

def make_bad_column(img, generator) -> torch.tensor:
    """Simulates 1 bad column within a given image.

    Args:
        img: (B, height, width) tensor, initial unmasked image
        generator: torch.Generator, random number generator used for column placement

    Returns:
        (B, height, width) tensor, the now "masked" image

    Remarks
    * Assumes that maximum indices for the height and width for the image are defined
      by img.height and img.width, prior to an added offset

    """

    # get the randomly choosen column
    column = torch.randint(img.shape[2] + 1, (1,), generator=generator,device=generator.device)

    # get the central pixel and 
    # the length the column will go from the central pixel
    center = torch.randint(img.shape[1] + 1, (1,), generator=generator,device=generator.device)
    length = torch.randint(torch.round(torch.div(img.shape[1], 0.5)).int(), (1,), generator=generator,device=generator.device)

    # determine min and max heights of the column
    # replace out-of-bounds indices with min/max in-bound values
    min_column = center - length
    min_column = torch.where(min_column >= 0, min_column, 0)

    max_column = center + length
    max_column = torch.where(max_column < img.shape[1], max_column, img.shape[1])

    img[:, min_column:max_column, column] = 0
    
    return img