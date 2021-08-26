from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


def peturb_bbox(bbox, pf=0, jf=0):
    '''
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[1] -= (pf*bheight) + (1-2*np.random.random())*jf*bheight
    pet_bbox[2] += (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[3] += (pf*bheight) + (1-2*np.random.random())*jf*bheight

    return pet_bbox


def square_bbox(bbox):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))
    
    dw_b_2 = int(round((maxdim-bwidth)/2.0))
    dh_b_2 = int(round((maxdim-bheight)/2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1
    
    return sq_bbox

    
def crop(img, bbox, bgval=0):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image        
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]
    
    img_out = np.ones((bheight, bwidth, nc))*bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2]+1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3]+1)
    
    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out

def compute_dt(mask,iters=10):
    """
    Computes distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation
    if iters>1:
        mask = binary_dilation(mask.copy(),iterations=iters)
    dist = distance_transform_edt(1-mask) / max(mask.shape)
    return dist

def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(1-mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1. / (1 + np.exp(k * -dist_diff))
    return dist

def sample_contour(
        mask,
    ):
        from skimage import measure
        # indices_y, indices_x = np.where(mask)
        # npoints = len(indices_y)
        contour = measure.find_contours(mask, 0)
        contour = np.concatenate(contour)
        sample_size = 1000

        def offset_and_clip_contour(contour, offset, img_size):
            contour = contour + offset
            contour = np.clip(contour, a_min=0, a_max=img_size - 1)
            return contour

        offsets = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, -1],
                [0, -2],
                [1, 0],
                [2, 0],
                [-1, 0],
                [-2, 0],
                [-1, -1],
                [-2, -2],
                [1, 1],
                [2, 2],
                [-1, 1],
                [-2, 2],
                [1, -1],
                [2, -2],
            ]
        )

        img_size = mask.shape[0]
        new_contours = []
        for offset in offsets:
            temp_contour = offset_and_clip_contour(
                contour, offset.reshape(-1, 2), img_size
            )
            new_contours.append(temp_contour)

        new_contours = np.concatenate(new_contours)
        # contour_mask = mask * 0
        # new_contours = new_contours.astype(np.int)
        # contour_mask[new_contours[:,0], new_contours[:,1]] = 1
        npoints = len(new_contours)
        sample_indices = np.random.choice(
            range(npoints), size=sample_size, replace=False
        )

        # swtich x any y.

        temp = np.stack(
            [new_contours[sample_indices, 1], new_contours[sample_indices, 0]],
            axis=1
        )
        temp = temp.copy()
        return temp

