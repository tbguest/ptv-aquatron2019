#!/usr/bin/env python

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sys

# import pandas as pd

%matplotlib qt5


# use ffmpeg to extract frames from video:
# ffmpeg -i pi73_1554902491.h264 -qscale:v 2 C:\\Projects\ptv-aquatron2019\data\
#             raw\Video\pi73\April10\extract_angle\pi73_1554902491_%05d.jpg -hide_banner


homechar = "C:\\"

# day = "Apr10"
day = "April11"

# pinum = "pi71"
pinum = "pi73"
# fname = "pi73_1554902491_00030.jpg"

imgdir = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "interim", \
                      day, "images", "calib", pinum)

imgs = sorted(glob.glob(os.path.join(imgdir, '*.jpg')))

plens = []

for img in imgs:

    im = plt.imread(img)

    plt.figure(1).clf()
    plt.imshow(im)
    plt.tight_layout()
    # plt.draw()
    plt.show()

    x = plt.ginput(2)

    plen = np.sqrt((x[0][1] - x[1][1])**2 + (x[0][0] - x[1][0])**2)
    plens.append(plen)

sticklen = 1.35 # m
scalings = sticklen/np.array(plens)

scaling = np.mean(scalings)
std_scaling = np.std(scalings)
