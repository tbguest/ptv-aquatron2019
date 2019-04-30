#!/usr/bin/env python

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
# from .data.regresstools import lowess
from datetime import datetime
import time
import pickle
import math
# import pandas as pd

%matplotlib qt5


# use ffmpeg to extract frames from video:
# ffmpeg -i pi73_1554902491.h264 -qscale:v 2 C:\\Projects\ptv-aquatron2019\data\
#             raw\Video\pi73\April10\extract_angle\pi73_1554902491_%05d.jpg -hide_banner


homechar = "C:\\"

day = "Apr10"
# day = "Apr11"

pinum = "pi71"
fname = "pi71_1554902488_00030.jpg"

# pinum = "pi73"
# fname = "pi73_1554902491_00030.jpg"

imgfn = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "raw", "Video", \
                      pinum, day, "extract_angle", fname)

# savedir = os.path.join(imgdir, "binary", "gaussian_blur")

# imgs = sorted(glob.glob(os.path.join(imgdir, '*.jpg')))

im = plt.imread(imgfn)

plt.figure(1).clf()
plt.imshow(im)
plt.tight_layout()
# plt.draw()
plt.xlim([1400, im.shape[1]])
plt.ylim([1000, 800])
# plt.xlim([0, 250])
# plt.ylim([1000, 900])
plt.show()

x = plt.ginput(2)

x

# compute angle:
jet_angle = math.atan((x[0][1] - x[1][1])/(x[1][0] - x[0][0]))

jet_angle*180/np.pi
