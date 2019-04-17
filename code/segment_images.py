import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import errno
import cv2 as cv


%matplotlib qt5

homechar = "C:\\"

# day = "April10"
day = "April11"

imgdir = os.path.join(homechar, "Projects", "Aquatron2019_April", "data", "interim", \
                      day, "images", "select")

savedir = os.path.join(imgdir, "binary", "gaussian_blur")

imgs = sorted(glob.glob(os.path.join(imgdir, '*.jpg')))


# main loop
for file in imgs:

    im = plt.imread(file)

    # im = plt.imread(imgs[199])


    # plt.figure(1).clf()
    # plt.imshow(im)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()

    ###################


    # gaussian filter:
    im_blur = cv.GaussianBlur(im,(5,5),0)

    # plt.figure(77).clf()
    # plt.imshow(blur)
    # plt.tight_layout()
    # plt.draw()
    # plt.show()

    r, g, b = cv.split(im_blur)

    # mask white
    minclr = 130
    is_blu = cv.inRange(b, minclr, 255)
    is_grn = cv.inRange(g, minclr, 255)
    is_red = cv.inRange(r, minclr, 255)
    wht_mask = np.logical_and(is_red, is_grn)
    wht_mask = np.logical_and(wht_mask, is_blu)

    nowhite = im_blur.copy()
    nowhite[wht_mask!=0] = (0,0,0)
    #
    # plt.figure(2)
    # plt.imshow(nowhite)
    # plt.tight_layout()


    is_blu = cv.inRange(b, 0, 255)
    is_grn = cv.inRange(g, 0, 255)
    is_red = cv.inRange(r, 0, 25)
    gb_mask = np.logical_and(is_red, is_grn)
    gb_mask = np.logical_and(gb_mask, is_blu)

    nogb = im_blur.copy()
    nogb[gb_mask!=0] = (0,0,0)

    nogbw = nogb.copy()
    nogbw[wht_mask!=0] = (0,0,0)

    # plt.figure(3)
    # plt.imshow(nogbw)
    # plt.tight_layout()

    nogbw[nogbw > 0] = 255

    fn = file[-25:-4] + '.jpg'

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.imsave(os.path.join(savedir, fn), nogbw)
