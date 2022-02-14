import math
import png
import struct
import array
import numpy as np
import cv2
import pdb

from io import *

UNKNOWN_FLOW_THRESH = 1e9;
UNKNOWN_FLOW = 1e10;

# Middlebury checks
TAG_STRING = 'PIEH'    # use this when WRITING the file
TAG_FLOAT = 202021.25  # check for this when READING the file

def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def save_pfm(file, image, scale = 1):
  import sys
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)


def ReadMiddleburyFloFile(path):
    """ Read .FLO file as specified by Middlebury.

    Returns tuple (width, height, u, v, mask), where u, v, mask are flat
    arrays of values.
    """

    with open(path, 'rb') as fil:
        tag = struct.unpack('f', fil.read(4))[0]
        width = struct.unpack('i', fil.read(4))[0]
        height = struct.unpack('i', fil.read(4))[0]

        assert tag == TAG_FLOAT
        
        #data = np.fromfile(path, dtype=np.float, count=-1)
        #data = data[3:]

        fmt = 'f' * width*height*2
        data = struct.unpack(fmt, fil.read(4*width*height*2))

        u = data[::2]
        v = data[1::2]

        mask = map(lambda x,y: abs(x)<UNKNOWN_FLOW_THRESH and abs(y) < UNKNOWN_FLOW_THRESH, u, v)
        mask = list(mask)
        u_masked = map(lambda x,y: x if y else 0, u, mask)
        v_masked = map(lambda x,y: x if y else 0, v, mask)

    return width, height, list(u_masked), list(v_masked), list(mask)

def ReadKittiPngFile(path):
    """ Read 16-bit .PNG file as specified by KITTI-2015 (flow).

    Returns a tuple, (width, height, u, v, mask), where u, v, mask
    are flat arrays of values.
    """
    # Read .png file.
    png_reader = png.Reader(path)
    data = png_reader.read()
    if data[3]['bitdepth'] != 16:
        raise Exception('bitdepth of ' + path + ' is not 16')

    width = data[0]
    height = data[1]

    # Get list of rows.
    rows = list(data[2])

    u = array.array('f', [0]) * width*height
    v = array.array('f', [0]) * width*height
    mask = array.array('f', [0]) * width*height

    for y, row in enumerate(rows):
        for x in range(width):
            ind = width*y+x
            u[ind] = (row[3*x] - 2**15) / 64.0
            v[ind] = (row[3*x+1] - 2**15) / 64.0
            mask[ind] = row[3*x+2]

            # if mask[ind] > 0:
            #     print(u[ind], v[ind], mask[ind], row[3*x], row[3*x+1], row[3*x+2])

    #png_reader.close()

    return (width, height, u, v, mask)


def WriteMiddleburyFloFile(path, width, height, u, v, mask=None):
    """ Write .FLO file as specified by Middlebury.
    """

    if mask is not None:
        u_masked = map(lambda x,y: x if y else UNKNOWN_FLOW, u, mask)
        v_masked = map(lambda x,y: x if y else UNKNOWN_FLOW, v, mask)
    else:
        u_masked = u
        v_masked = v

    fmt = 'f' * width*height*2
    # Interleave lists
    data = [x for t in zip(u_masked,v_masked) for x in t]

    with open(path, 'wb') as fil:
        fil.write(str.encode(TAG_STRING))
        fil.write(struct.pack('i', width))
        fil.write(struct.pack('i', height))
        fil.write(struct.pack(fmt, *data))


def write_flow(path,flow):
    
    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = flow[:, :, 0:2]*64.+ 2 ** 15
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    flow = flow.astype(np.uint16)
    flow = cv2.imwrite(path, flow[:,:,::-1])

    #WriteKittiPngFile(path,
    #     flow.shape[1], flow.shape[0], flow[:,:,0].flatten(), 
    #    flow[:,:,1].flatten(), flow[:,:,2].flatten())
    


def WriteKittiPngFile(path, width, height, u, v, mask=None):
    """ Write 16-bit .PNG file as specified by KITTI-2015 (flow).

    u, v are lists of float values
    mask is a list of floats, denoting the *valid* pixels.
    """

    data = array.array('H',[0])*width*height*3

    for i,(u_,v_,mask_) in enumerate(zip(u,v,mask)):
        data[3*i] = int(u_*64.0+2**15)
        data[3*i+1] = int(v_*64.0+2**15)
        data[3*i+2] = int(mask_)

        # if mask_ > 0:
        #     print(data[3*i], data[3*i+1],data[3*i+2])

    with open(path, 'wb') as png_file:
        png_writer = png.Writer(width=width, height=height, bitdepth=16, compression=3, greyscale=False)
        png_writer.write_array(png_file, data)


def ConvertMiddleburyFloToKittiPng(src_path, dest_path):
    width, height, u, v, mask = ReadMiddleburyFloFile(src_path)
    WriteKittiPngFile(dest_path, width, height, u, v, mask=mask)

def ConvertKittiPngToMiddleburyFlo(src_path, dest_path):
    width, height, u, v, mask = ReadKittiPngFile(src_path)
    WriteMiddleburyFloFile(dest_path, width, height, u, v, mask=mask)


def ParseFilenameKitti(filename):
    # Parse kitti filename (seq_frameno.xx),
    # return seq, frameno, ext.
    # Be aware that seq might contain the dataset name (if contained as prefix)
    ext = filename[filename.rfind('.'):]
    frameno = filename[filename.rfind('_')+1:filename.rfind('.')]
    frameno = int(frameno)
    seq = filename[:filename.rfind('_')]
    return seq, frameno, ext


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_calib_cam_to_cam(cam_to_cam_file):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_file)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
    P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    data['b00'] = P_rect_00[0, 3] / P_rect_00[0, 0]
    data['b10'] = P_rect_10[0, 3] / P_rect_10[0, 0]
    data['b20'] = P_rect_20[0, 3] / P_rect_20[0, 0]
    data['b30'] = P_rect_30[0, 3] / P_rect_30[0, 0]

    return data

