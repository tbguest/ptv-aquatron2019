
# coding: utf-8

# # Plot cork trajectory data

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
# from .data.regresstools import lowess
from datetime import datetime
import time
import pickle
import pandas as pd
import trackpy as tp

%matplotlib qt5


# use ffmpeg to decimate video files:


homechar = "C:\\"

# # 70% flow
# vid = "pi71_1554992783"
# vid = "pi73_1554992784"
vid = "pi71_1554993824" # yes
# vid = "pi73_1554993823"
# # 80% flow
# vid = "pi71_1554994358" # yes
# vid = "pi73_1554994357"
# vid = "pi71_1554994823" # yes
# vid = "pi73_1554994822"
# # 90% flow
# vid = "pi71_1554995860" # yes
# vid = "pi73_1554995859"
# vid = "pi71_1554996417"
# vid = "pi73_1554996416"
# # 100% flow
# vid = "pi71_1554997267"
# vid = "pi73_1554997266"
# vid = "pi71_1554997692" # yes
# vid = "pi73_1554997690"
# vid = "pi71_1554997998" # yes - long deployment
# vid = "pi73_1554997995"

flow = "flow70"
# flow = "flow80"
# flow = "flow90"
# flow = "flow100"

pinum = vid[2:4]

# input file
inptfile = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "processed", vid + ".pkl")
infile = open(inptfile,'rb')
data = pickle.load(infile)
infile.close()

## initialize empty dataframe with all fields in place (might be better way to do this)
# df0 = data[data['particle'] == -1]
df0 = pd.DataFrame()
pathlen = []
pmax = np.max(data['particle'].values)

# initialize grid
nx = 1640
ny = 1232
dn = 10

# scaling [from extract_scaling.py]
scaling = 0.003454318809170935 # m/pix

# # Rotate coordinates
# approximate location of jet outflow (on rotation axis)
# angles and new origin computed in `extract_jetangle.py`
if pinum == '71':
    new_originx = 1640
    new_originy = 940
    theta_jet = np.pi + 0.2682013644206911 # rad
else:
    new_originx = 1640
    new_originy = 940
    theta_jet = np.pi + -0.11809202841909477 # rad

# rotate data
R = [[np.cos(theta_jet), np.sin(theta_jet)], [-np.sin(theta_jet), np.cos(theta_jet)]]
data_p = data.copy()
xvec = np.squeeze([data_p['x'].values - new_originx, data_p['y'].values - new_originy])
xvec_rot = np.matmul(R, xvec)
xpr = pd.DataFrame({'frame': data_p['frame'], 'xp': xvec_rot[0]})
ypr = pd.DataFrame({'frame': data_p['frame'], 'yp': xvec_rot[1]})

# this adds an additional 'frames' column. Not sure how to avoid this
# took a while to maintain indices of rotated coordinates
data_rot = pd.concat([data_p, ypr], axis=1)# sort=False
data_rot = pd.concat([data_rot, xpr], axis=1)# sort=False

# delete duplicate 'frame' columns
data_rot = data_rot.loc[:,~data_rot.columns.duplicated()]

## initialize empty dataframe with all fields in place (might be better way to do this)
# df0 = data[data['particle'] == -1]
df0 = pd.DataFrame()

# init fig
# fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='redraw trajectories')
# ax1.set_xlabel('axial jet coordinate [m]')
# ax1.set_ylabel('transverse jet coordinate [m]')

fig1 = plt.figure(num='redraw trajectories', figsize=(4,6))
ax11 = plt.subplot(211)
ax11.set_xlabel('x [m]')
ax11.set_ylabel('y [m]')

# iterate over particle ids
for i in range(0, pmax):

    # organize by particle
    df = data_rot[data_rot['particle'] == i]

    # if the particle exists (has data associated)
    if not df.empty:

        pathlen.append(len(df))

        # trajectory plot
        fig1.gca()
        ax11.plot(df['xp'].values*scaling, df['yp'].values*scaling)

        # compute difference
        dfjnk = df.groupby('particle').diff()#/df['frame'].values
        dx_rot = dfjnk['xp']*scaling
        dy_rot = dfjnk['yp']*scaling
        dframe = dfjnk['frame']

        # velocity components
        u_rot = dx_rot/dframe
        v_rot = dy_rot/dframe
        uvel_rot = pd.DataFrame({'u': u_rot})
        vvel_rot = pd.DataFrame({'v': v_rot})

        # join velocities with temp dataframe, then append to new dataframe
        dfvel_rot = df.join(uvel_rot)
        dfvel_rot = dfvel_rot.join(vvel_rot)
        df0 = df0.append(dfvel_rot)


# boundaries of new grid
minx = np.min(df0['xp'].values).astype(int)
maxx = np.max(df0['xp'].values).astype(int)
miny = np.min(df0['yp'].values).astype(int)
maxy = np.max(df0['yp'].values).astype(int)

# define new grid
xgrid = np.arange(minx, maxx, dn)
ygrid = np.arange(miny, maxy, dn)
xxgrid, yygrid = np.meshgrid(xgrid, ygrid)
ugrid = np.zeros(np.shape(xxgrid))
vgrid = np.zeros(np.shape(yygrid))
gridcount = np.zeros(np.shape(xxgrid))

# loop through all entries and append velocity to relevant bin
for j in range(0, len(df0)):
    if not np.isnan(df0['u'].values[j]):
        Iy = ((df0['yp'].values[j] - miny)/dn).astype(int)
        Ix = ((df0['xp'].values[j] - minx)/dn).astype(int)
        ugrid[Iy-1, Ix-1] += df0['u'].values[j]
    if not np.isnan(df0['v'].values[j]):
        Iy = ((df0['yp'].values[j] - miny)/dn).astype(int)
        Ix = ((df0['xp'].values[j] - minx)/dn).astype(int)
        vgrid[Iy-1, Ix-1] += df0['v'].values[j]
        # assume that if ~isnan u then ~isnan v:
        gridcount[Iy-1, Ix-1] += 1.


umean = np.divide(ugrid.astype(float),gridcount.astype(float))
vmean = np.divide(vgrid.astype(float),gridcount.astype(float))

# for colorscale
maxu = np.nanmax(umean[:])
minu = np.nanmin(umean[:])
maxv = np.nanmax(vmean[:])
minv = np.nanmin(vmean[:])
clrmax = np.max([np.abs(maxu), np.abs(minu), np.abs(maxv), np.abs(minv)])
clrmax = clrmax - 0.25*(clrmax)



### PLOTTING ####

# set lims on traj plot
fig1.gca()
ax11.set_ylim([miny*scaling, maxy*scaling])
ax11.set_xlim([minx*scaling, maxx*scaling])

# add hist to 2-panel plot
ax12 = plt.subplot(212)
ax12.hist(np.array(pathlen)*scaling, bins=30)
ax12.set_xlabel('path length [m]')
ax12.set_ylabel('count')
ax12.text(250*scaling, 250, 'trajectories=' + str(len(pathlen)))
fig1.tight_layout()


# velocity data
fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(9.3, 7.5), num='velocities')
im0 = ax2[0].imshow(umean, origin='lower', extent=(minx*scaling, maxx*scaling, miny*scaling, maxy*scaling), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
im1 = ax2[1].imshow(vmean, origin='lower', extent=(minx*scaling, maxx*scaling, miny*scaling, maxy*scaling), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
# fig.colorbar()
# plt.set_cmap('bwr')
cbar_ax = fig2.add_axes([0.8, 0.15, 0.05, 0.7])
clb = fig2.colorbar(im0, cax=cbar_ax)
ax2[1].set_xlabel('x [m]')
ax2[1].set_ylabel('y [m]')
ax2[0].set_ylabel('y [m]')
clb.ax.set_title('speed [m/s]')
ax2[0].text(1., 1.5, 'u component')
ax2[1].text(1., 1.5, 'v component')


saveFlag = 1
if saveFlag is 1:
    savedn = os.path.join('C:\\','Projects','ptv-aquatron2019','reports','figures', flow, vid)
    if not os.path.exists(savedn):
        try:
            os.makedirs(savedn)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig1.savefig(os.path.join(savedn, 'trajectories_and_histogram.png'), dpi=1000, transparent=True)
    fig1.savefig(os.path.join(savedn, 'trajectories_and_histogram.pdf'), dpi=None, transparent=True)

    fig2.savefig(os.path.join(savedn, 'mean_velocity.png'), dpi=1000, transparent=True)
    fig2.savefig(os.path.join(savedn, 'mean_velocity.pdf'), dpi=None, transparent=True)
