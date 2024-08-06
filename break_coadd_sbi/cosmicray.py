"""Method to randomly generate a simple cosmic ray on an image.
"""

import torch

def generate_cosmic_ray(img, generator) -> torch.tensor:
    """Simulates 1 cosmic ray within a given image.

    Args:
        img: (B, height, width) tensor, initial unmasked image
        generator: torch.Generator, random number generator used for column placement

    Returns:
        (B, height, width) tensor, the now "masked" image

    """

    # choose two randomly choosen grid points
    # to act as the end points of the cosmic ray
    initial_points_xs = torch.randint(img.shape[2], (2,), generator=generator,device=generator.device)
    initial_points_ys = torch.randint(img.shape[1], (2,), generator=generator,device=generator.device)

    # define the start point as the point with a smaller x value (for simplicity later)
    if initial_points_xs[1]>initial_points_xs[0]:
        start_point = torch.tensor([initial_points_xs[0],initial_points_ys[0]])
        end_point = torch.tensor([initial_points_xs[1],initial_points_ys[1]])
    elif initial_points_xs[1]<initial_points_xs[0]:
        start_point = torch.tensor([initial_points_xs[1],initial_points_ys[1]])
        end_point = torch.tensor([initial_points_xs[0],initial_points_ys[0]])
    # vertical lines must be treated differently since their slope is undefined
    else: # initial_points_xs[0]==initial_points_xs[1]
        # determine which points are above/below
        if initial_points_ys[1]>initial_points_ys[0]:
            start_point = torch.tensor([initial_points_xs[0],initial_points_ys[0]])
            end_point = torch.tensor([initial_points_xs[1],initial_points_ys[1]])
        else:
            start_point = torch.tensor([initial_points_xs[1],initial_points_ys[1]])
            end_point = torch.tensor([initial_points_xs[0],initial_points_ys[0]])
        
        cr_ys = torch.arange(start_point[1], end_point[1] + 1)
        cr_xs = torch.arange(start_point[0], end_point[0] + 1)

        cr_coords = torch.tensor([[start_point[0], cr_ys[i]] for i in range(len(cr_ys))]).to(torch.long)
        img[:, cr_coords[:, 0], cr_coords[:, 1]] = 0
        return img

    # determine the slope of the line from the two points
    slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

    # determine the intercept using the start point as a test point
    intercept = start_point[1] - slope * start_point[0]

    # create tensor for all x's on grid b/w and including start and end points
    cr_xs = torch.arange(start_point[0], end_point[0] + 1)
    
    # caluclate the associated y value for each x in the tensor
    # exact y values are then rounded to land on nearest grid coordinate
    cr_ys = torch.round(torch.add(torch.mul(cr_xs, slope), intercept))

    # remove rounded y values and associated x values that go over index amount
    if cr_ys.max() >= 40:
        cr_xs = cr_xs[cr_ys < 40]
        cr_ys = cr_ys[cr_ys < 40]

    # TO-DO: implement algorithm here to fill in corner pixels
    # likely as an additional function

    # combine the two coordinate tensors into a single tensor
    cr_coords = torch.tensor([[x, cr_ys[i]] for i, x in enumerate(cr_xs)]).to(torch.long)

    # set the cosmic ray pixels in the input image to 0
    img[:, cr_coords[:, 1], cr_coords[:, 0]] = 0
    return img