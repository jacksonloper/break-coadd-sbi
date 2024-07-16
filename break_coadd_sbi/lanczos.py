"""
Lanczos interpolation with a=2 for PyTorch.
"""

import torch

def _get_slice_unsafe(tensor, start, length, axis):
    idxs = torch.arange(length) + start
    return torch.index_select(tensor, axis, idxs)


def _get_slices_unsafe(tensor, starts, length, axis):
    """
    Args:
        tensor: a tensor of shape [B, ...]
        starts: a tensor of shape [B] containing the start points for each slice
        length: the length of each slice
        axis: the axis along which to slice (must be at least 1, we refuse to slice along batch axis)

    Returns:
        sliced tensor

    If axis=1, this is the same as

        return[i] = tensor[i, starts[i]:starts[i]+length, ...]

    If axis=2, this is the same as

        return[i] = tensor[i, :, starts[i]:starts[i]+length, ...]

    etc.

    If starts are negative or lengths exceed the dimension, the function will fail,
    perhaps silently.  It is up to the user to ensure that the inputs are valid.
    """

    batched = torch.vmap(_get_slice_unsafe, (0, 0, None, None))
    return batched(tensor, starts, length, axis - 1)


def get_slices(tensor, starts, length, axis):
    r"""
    Args:
        tensor: a tensor of shape [B, ...]
        starts: a tensor of shape [B] containing the start points for each slice
        length: the length of each slice
        axis: the axis along which to slice (must be at least 1, doesn't make sense to slice along batch axis)

    Returns:
         a new tensor formed by slicing each row of the old tensor along the specified
         axis with the corresponding start point.

    If axis=1, this is akin to

        return[i] = tensor[i, starts[i]:starts[i]+length, ...]

    If axis=2, this is akin to

        return[i] = tensor[i, :, starts[i]:starts[i]+length, ...]

    etc.

    This function allows great flexibility where the slicing is concerned
    * starts are permitted to be negative
    * lengths are permitted to exceed the dimension
    * no interpolation is used to fill in the gaps (starts must be integers)

    Returns
    * a new tensor of shape [B, length, ...]
    * a footprint tensor of the same shape as the sliced tensor,
        with 1s where the new tensor is valid
    """

    # enforce that axis is at least 1
    if axis < 1:
        raise ValueError("Axis must be at least 1.")

    # insist on same batch dimension
    if tensor.shape[0] != starts.shape[0]:
        raise ValueError("Tensor and starts must have the same batch dimension.")

    # figure out worst case padding on the left (but not negative)
    left_pad = torch.clamp(-starts.min(), 0)
    # and the right
    right_pad = torch.clamp(starts.max() + length - tensor.shape[axis], 0)

    # do the pad on the corresponding axis
    # remember that pad expects a tuple of length 2*dim
    # and the tuples are ordered from the last dimension
    # to the first
    # if ndim = 4, and axis = 3, then we place paddings at 0 and 1
    # if ndim = 4, and axis = 2, then we place paddings at 2 and 3
    axis_reverse_order = tensor.ndim - 1 - axis
    paddings = [0] * tensor.ndim * 2
    paddings[2 * axis_reverse_order] = left_pad
    paddings[2 * axis_reverse_order + 1] = right_pad
    padded_tensor = torch.nn.functional.pad(tensor[None], paddings,mode='reflect')[0]

    # adjust the starts based on our padding
    starts = starts + left_pad

    # get footprint, by getting slices of a synthetic tensor of ones
    onetensor = torch.ones_like(tensor)
    onetensor_padded = torch.nn.functional.pad(onetensor, paddings)
    footprint = _get_slices_unsafe(onetensor_padded, starts, length, axis)

    # and for the real data, we can use the same function
    result = _get_slices_unsafe(padded_tensor, starts, length, axis)

    # done
    return result, footprint


def _batchconv_H(imgs, filters):
    '''
    Args:
        imgs: (B, H, W) tensor
        filters: (B, 5) tensor of filters

    Returns:
        (B, H, W) tensor of convolved images

    out[b, i, j] = sum_k filters[b, k] * imgs[b, i + k - 2, j]
    '''

    # pad the H part of the images
    imgs = torch.nn.functional.pad(imgs, (0, 0, 2, 2),mode='reflect')

    # unroll the convolution
    out = imgs[:,:-4]*filters[:,0][:,None,None]
    out = out + imgs[:,1:-3]*filters[:,1][:,None,None]
    out = out + imgs[:,2:-2]*filters[:,2][:,None,None]
    out = out + imgs[:,3:-1]*filters[:,3][:,None,None]
    out = out + imgs[:,4:]*filters[:,4][:,None,None]

    return out

def _batchconv_W(imgs, filters):
    '''
    Args:
        imgs: (B, H, W) tensor
        filters: (B, 5) tensor of filters

    Returns:
        (B, H, W) tensor of convolved images

    out[b, i, j] = sum_k filters[b, k] * imgs[b, i, j + k - 2]
    '''

    # pad the H part of the images
    imgs = torch.nn.functional.pad(imgs, (2, 2),mode='reflect')

    # unroll the convolution
    out = imgs[:,:,:-4]*filters[:,0][:,None,None]
    out = out + imgs[:,:,1:-3]*filters[:,1][:,None,None]
    out = out + imgs[:,:,2:-2]*filters[:,2][:,None,None]
    out = out + imgs[:,:,3:-1]*filters[:,3][:,None,None]
    out = out + imgs[:,:,4:]*filters[:,4][:,None,None]

    return out


def lanczos_interp_large(imgs, shifts):
    '''
    Args:
        imgs: (B, H, W) tensor
        shifts: (B, 2) tensor of shifts

    Returns:
        (B, H, W) tensor of interpolated images
    '''

    ishifts = torch.round(shifts)
    fshifts = shifts-ishifts
    ishifts = ishifts.int()

    imgs = lanczos_interp(imgs,fshifts)
    imgs,_ = get_slices(imgs, -ishifts[:,0], imgs.shape[1], 1)
    imgs,_ = get_slices(imgs, -ishifts[:,1], imgs.shape[2], 2)

    return imgs

def lanczos_interp(imgs, shifts):
    '''
    Args:
        imgs: (B, H, W) tensor
        shifts: (B, 2) tensor of shifts, should be between -.5 and .5

    Returns:
        (B, H, W) tensor of interpolated images
    '''

    if torch.abs(shifts).max().item()>.5:
        raise Exception("Shifts are too big")

    # for each shift, we need to construct a lanczos filter
    # with a=2
    # construct 1d lanczos filter with a=2
    # final filter information has shape (B,5,2)
    x = torch.arange(-2, 3, device=imgs.device).float()[None,:,None] + shifts[:,None,:]
    lanczos = torch.sinc(x) * torch.sinc(x / 2)
    lanczos /= lanczos.sum(axis=1)[:,None,:]

    # apply the filters
    imgs = _batchconv_H(imgs, lanczos[:,:,0])
    imgs = _batchconv_W(imgs, lanczos[:,:,1])

    return imgs