"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import png
from flowutils.util_flow import readPFM
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pdb


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

"""
=============
Flow Section
=============
"""


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def point_vec(img,flow,skip=40):
    maxsize=1000.
    extendfac=1.
    resize_factor = 1
    #resize_factor = max(1,int(max(maxsize/img.shape[0], maxsize/img.shape[1])))
    meshgrid = np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    dispimg = cv2.resize(img[:,:,::-1].copy(), None,fx=resize_factor,fy=resize_factor)
    colorflow = flow_to_image(flow).astype(int)
    for i in range(img.shape[1]): # x 
        for j in range(img.shape[0]): # y
            if flow[j,i,2] != 1: continue
            if j%skip!=0 or i%skip!=0: continue
            xend = int((meshgrid[0][j,i]+extendfac*flow[j,i,0])*resize_factor)
            yend = int((meshgrid[1][j,i]+extendfac*flow[j,i,1])*resize_factor)
            leng = np.linalg.norm(flow[j,i,:2]*extendfac)
            if leng<1:continue
            dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j,i]*resize_factor,meshgrid[1][j,i]*resize_factor),\
                                      (xend,yend),
                                      (int(colorflow[j,i,2]),int(colorflow[j,i,1]),int(colorflow[j,i,0])),1,tipLength=4/leng,line_type=cv2.LINE_AA)
    return dispimg


def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = np.sqrt(du**2+dv**2).max()
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.fmod(np.arctan2(dv, du) / (2 * np.pi)+1.,1.)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid

    return img


def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    elif filename.endswith('.png'):
        flow = read_png_file(filename)
    elif filename.endswith('.pfm'):
        flow = read_pfm_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow

import numpy as np
import os


def write_flo(flow, filename):

    TAG_STRING = b'PIEH'
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]

    height, width, nBands = flow.shape
    assert nBands == 2, "Number of bands = %r != 2" % nBands
    u = flow[: , : , 0]
    v = flow[: , : , 1] 
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    f = open(filename,'wb')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def save_flow_image(flow, image_file):
    """
    save flow visualization into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    flow_img = flow_to_image(flow)
    img_out = Image.fromarray(flow_img)
    img_out.save(image_file)


def flowfile_to_imagefile(flow_file, image_file):
    """
    convert flowfile into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    flow = read_flow(flow_file)
    save_flow_image(flow, image_file)


def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg


def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mepe


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def evaluate_flow_file(gt_file, pred_file):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt_file: ground truth file path
    :param pred_file: estimated optical flow file path
    :return: end point error, float32
    """
    # Read flow files and calculate the errors
    gt_flow = read_flow(gt_file)        # ground truth flow
    eva_flow = read_flow(pred_file)     # predicted flow
    # Calculate errors
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe


"""
==============
Disparity Section
==============
"""


def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) / w
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for i in range(len(image_data)):
        for j in range(channel):
            flow[i, :, j] = image_data[i][j::channel]
    return flow[:, :, 0] / 256


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = disp.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((disp, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()


"""
==============
Image Section
==============
"""


def read_image(filename):
    """
    Read normal image of any format
    :param filename: name of the image file
    :return: image data in matrix uint8 type
    """
    img = Image.open(filename)
    im = np.array(img)
    return im


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:,:,0]
    fy += flow[:,:,1]
    mask = np.logical_or(fx <0 , fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        plt.imshow(channel, cmap='gray')
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


"""
==============
Others
==============
"""

def pfm_to_flo(pfm_file):
    flow_filename = pfm_file[0:pfm_file.find('.pfm')] + '.flo'
    (data, scale) = readPFM(pfm_file)
    flow = data[:, :, 0:2]
    write_flow(flow, flow_filename)


def scale_image(image, new_range):
    """
    Linearly scale the image into desired range
    :param image: input image
    :param new_range: the new range to be aligned
    :return: image normalized in new range
    """
    min_val = np.min(image).astype(np.float32)
    max_val = np.max(image).astype(np.float32)
    min_val_new = np.array(min(new_range), dtype=np.float32)
    max_val_new = np.array(max(new_range), dtype=np.float32)
    scaled_image = (image - min_val) / (max_val - min_val) * (max_val_new - min_val_new) + min_val_new
    return scaled_image.astype(np.uint8)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        #print("Reading %d x %d flow file in .flo format" % (h, w))
        flow = np.ones((h[0],w[0],3))
        data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
        flow[:,:,:2] = data2d
    f.close()
    return flow


def read_png_file(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow = cv2.imread(flow_file,-1)[:,:,::-1].astype(np.float64)
 #   flow_object = png.Reader(filename=flow_file)
 #   flow_direct = flow_object.asDirect()
 #   flow_data = list(flow_direct[2])
 #   (w, h) = flow_direct[3]['size']
 #   #print("Reading %d x %d flow file in .png format" % (h, w))
 #   flow = np.zeros((h, w, 3), dtype=np.float64)
 #   for i in range(len(flow_data)):
 #       flow[i, :, 0] = flow_data[i][0::3]
 #       flow[i, :, 1] = flow_data[i][1::3]
 #       flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow


def read_pfm_file(flow_file):
    """
    Read from .pfm file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    (data, scale) = readPFM(flow_file)
    return data 


# fast resample layer
def resample(img, sz):
    """
    img: flow map to be resampled
    sz: new flow map size. Must be [height,weight]
    """
    original_image_size = img.shape
    in_height = img.shape[0]
    in_width = img.shape[1]
    out_height = sz[0]
    out_width = sz[1]
    out_flow = np.zeros((out_height, out_width, 2))
    # find scale
    height_scale =  float(in_height) / float(out_height)
    width_scale =  float(in_width) / float(out_width)

    [x,y] = np.meshgrid(range(out_width), range(out_height))
    xx = x * width_scale
    yy = y * height_scale
    x0 = np.floor(xx).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(yy).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0,0,in_width-1)
    x1 = np.clip(x1,0,in_width-1)
    y0 = np.clip(y0,0,in_height-1)
    y1 = np.clip(y1,0,in_height-1)

    Ia = img[y0,x0,:]
    Ib = img[y1,x0,:]
    Ic = img[y0,x1,:]
    Id = img[y1,x1,:]

    wa = (y1-yy) * (x1-xx)
    wb = (yy-y0) * (x1-xx)
    wc = (y1-yy) * (xx-x0)
    wd = (yy-y0) * (xx-x0)
    out_flow[:,:,0] = (Ia[:,:,0]*wa + Ib[:,:,0]*wb + Ic[:,:,0]*wc + Id[:,:,0]*wd) * out_width / in_width
    out_flow[:,:,1] = (Ia[:,:,1]*wa + Ib[:,:,1]*wb + Ic[:,:,1]*wc + Id[:,:,1]*wd) * out_height / in_height

    return out_flow

