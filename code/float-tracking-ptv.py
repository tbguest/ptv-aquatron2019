
# coding: utf-8


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import scipy.io
import glob
import os
import errno

# packages for image manipulation and tracking (will need to be installed on local machine)
import pims
import trackpy as tp
import cv2 as cv


# see ipynb of the same name for a comprehensive walkthrough

# function that segments out the red-topped corks:
def segmentImage(file):

    im = plt.imread(file)

    # gaussian filter:
    im_blur = cv.GaussianBlur(im,(5,5),0)

    # split colour channels
    r, g, b = cv.split(im_blur)

    # mask white
    minclr = 130
    is_blu = cv.inRange(b, minclr, 255)
    is_grn = cv.inRange(g, minclr, 255)
    is_red = cv.inRange(r, minclr, 255)
    wht_mask = np.logical_and(is_red, is_grn)
    wht_mask = np.logical_and(wht_mask, is_blu)

    # grn/blu mask
    is_blu = cv.inRange(b, 0, 255)
    is_grn = cv.inRange(g, 0, 255)
    is_red = cv.inRange(r, 0, 25)
    gb_mask = np.logical_and(is_red, is_grn)
    gb_mask = np.logical_and(gb_mask, is_blu)

    # apply masks
    nogb = im_blur.copy()
    nogb[gb_mask!=0] = (0,0,0)
    nogbw = nogb.copy()
    nogbw[wht_mask!=0] = (0,0,0)

    # "turn up" all non-zero values
    nogbw[nogbw > 0] = 255

    return nogbw



# Example code to extract all frames at highst quality:`ffmpeg -i pi71_1554994358.h264 -qscale:v 2 C:\Projects\Aquatron2019_April\data\interim\April11\images\pi71_1554994358_%05d.jpg -hide_banner`

homechar = "C:\\"

# day = "April10"
day = "April11"

# vid = "pi71_1554994358" # ?
# vid = "pi71_1554994823" # 70% flow?
# vid = "pi71_1554995860" # 70?

# 100%
vid = "pi71_1554997692"


imgdir = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "interim", \
        day, "images", vid, "select")
savedir = os.path.join(imgdir, "binary")

if not os.path.exists(savedir):
    try:
        os.makedirs(savedir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

imgs = sorted(glob.glob(os.path.join(imgdir, '*.jpg')))
binimgs = sorted(glob.glob(os.path.join(savedir, '*.jpg')))

# check if segmentation step has already been done
if len(binimgs) == 0:
    # Apply the segmentation to all the images. This takes a while (10s of minutes).
    for file in imgs:
        nogbw = segmentImage(file)
        fn = file[-25:-4] + '.jpg'
        plt.imsave(os.path.join(savedir, fn), nogbw)

# define pims image sequence
frames = pims.ImageSequence(os.path.join(savedir, '*.jpg'), as_grey=True)

# trackpy calls (takes a while)
d_cork = 11 # estimated diameter of tracked features
f = tp.batch(frames, d_cork, minmass=0, invert=False)
t = tp.link_df(f, 9, memory=5)
t1 = tp.filter_stubs(t, 50)

# save in python and matlab readable formats
scipy.io.savemat(os.path.join(homechar, "Projects", "Aquatron2019_April", "data", \
        "processed", vid + ".mat"), {'cork_traj':t1.to_dict("list")})
t1.to_pickle(os.path.join(homechar, "Projects", "Aquatron2019_April", "data", \
        "processed", vid + ".pkl"))  # where to save it, usually as a .pkl
