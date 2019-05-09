
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


homechar = "C:\\"

# day = "April10"
day = "April11"

vid = "pi71_1554994358" # ?
# vid = "pi71_1554994823" # 70% flow?
# vid = "pi71_1554995860" # 70?

inptfile = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "processed", vid + ".pkl")

infile = open(inptfile,'rb')
data = pickle.load(infile)
infile.close()

## initialize empty dataframe with all fields in place (might be better way to do this)
# df0 = data[data['particle'] == -1]
df0 = pd.DataFrame()

pathlen = []

pmax = np.max(data['particle'].values)
# pmax = 10000

# initialize grid
nx = 1640
ny = 1232
# dn = 5
#
# for i in range(0, pmax):
#
#     # organize by particle
#     df2 = data[data['particle'] == i]
#
#     if not df2.empty:
#
#         pathlen.append(len(df2))
#
#         # compute difference
#         df3 = df2.groupby('particle').diff()#/df2['frame'].values
#         dx = df3['x']
#         dy = df3['y']
#         dframe = df3['frame']
#
#         # velocity components
#         u = dx/dframe
#         v = dy/dframe
#         uvel = pd.DataFrame({'u': u})
#         vvel = pd.DataFrame({'v': v})
#
#         # join velocities with temp dataframe, then append to new dataframe
#         dfvel = df2.join(uvel)
#         dfvel = dfvel.join(vvel)
#         df0 = df0.append(dfvel)
#

#
# xgrid = np.arange(0, nx+1, dn)
# ygrid = np.arange(0, ny+1, dn)
# xxgrid, yygrid = np.meshgrid(xgrid, ygrid)
# ugrid = np.zeros(np.shape(xxgrid))
# vgrid = np.zeros(np.shape(yygrid))
# gridcount = np.zeros(np.shape(xxgrid))
#
# # loop through all entries and append velocity to relevant bin
# for j in range(0, len(df0)):
#     if not np.isnan(df0['u'].values[j]):
#         ugrid[(df0['y'].values[j]/dn).astype(int), (df0['x'].values[j]/dn).astype(int)] += df0['u'].values[j]
#     if not np.isnan(df0['v'].values[j]):
#         vgrid[(df0['y'].values[j]/dn).astype(int), (df0['x'].values[j]/dn).astype(int)] += df0['v'].values[j]
#     gridcount[(df0['y'].values[j]/dn).astype(int), (df0['x'].values[j]/dn).astype(int)] += 1.
#
# umean = np.divide(ugrid.astype(float),gridcount.astype(float))
# vmean = np.divide(vgrid.astype(float),gridcount.astype(float))
#
#
# fig2 = plt.figure(num='trajectory data', figsize=(10,6))
# ax23 = plt.subplot(223)
# ax23.hist(pathlen, bins=30)
# ax23.set_xlabel('path length [pix]')
# ax23.set_ylabel('count')
# ax23.text(250, 250, 'trajectories=' + str(len(pathlen)))
#
# ax22 = plt.subplot(222)
# im22 = ax22.imshow(umean)
# # ax22.set_ylabel('y [pix]')
# ax22.text(25, 25, 'u component')
#
# # fig2.tight_layout()
#
# ax24 = plt.subplot(224)
# im24 = ax24.imshow(vmean)
# ax24.set_xlabel('x [pix]')
# ax24.set_ylabel('y [pix]')
# ax24.text(25, 25, 'v component')
#
#
# fig2.tight_layout()
#
# plt.set_cmap('bwr')
#
# fig2.subplots_adjust(right=0.8)
# cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
# clb = fig2.colorbar(im22, cax=cbar_ax)
# # plt.show()
# im22.set_clim(vmin=-6, vmax=6)
# im24.set_clim(vmin=-6, vmax=6)
# clb.ax.set_title('speed [m/s]')
#
# # fig2.tight_layout()
#
#
#
#
# # fig1 = plt.figure(figsize=(12,8), num='trajectories')
# ax21 = plt.subplot(221)
# tp.plot_traj(data, colorby='particle')
# # ax21.xlim([0, 1640])
# # ax21.ylim([1232, 0])
#
#
# fig2 = plt.figure(num='velocity', figsize=(5,6))
#
# ax22 = plt.subplot(211)
# im22 = ax22.imshow(umean)
# ax22.set_ylabel('y [pix]')
# ax22.text(25, 25, 'u component')
#
# ax24 = plt.subplot(212)
# im24 = ax24.imshow(vmean)
# ax24.set_xlabel('x [pix]')
# ax24.set_ylabel('y [pix]')
# ax24.text(25, 25, 'v component')
#
#
# fig2.tight_layout()
#
# plt.set_cmap('bwr')
#
# fig2.subplots_adjust(right=0.8)
# cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
# clb = fig2.colorbar(im22, cax=cbar_ax)
# # plt.show()
# im22.set_clim(vmin=-6, vmax=6)
# im24.set_clim(vmin=-6, vmax=6)
# clb.ax.set_title('speed [pix/s]')
#
# # fig2.tight_layout()
#
#
# fig3 = plt.figure(num='trajectories', figsize=(9,6))
# ax32 = plt.subplot(222)
# ax32.hist(pathlen, bins=30)
# ax32.set_xlabel('path length [pix]')
# ax32.set_ylabel('count')
# ax32.text(250, 250, 'trajectories=' + str(len(pathlen)))
#
# # fig1 = plt.figure(figsize=(12,8), num='trajectories')
# ax31 = plt.subplot(221)
# tp.plot_traj(data, colorby='particle')
# # ax21.xlim([0, 1640])
# # ax21.ylim([1232, 0])
#
# saveFlag = 0
# if saveFlag is 1:
#     savedn = os.path.join('C:\\','Projects','ptv-aquatron2019','reports','figures','trajectories', vid)
#     if not os.path.exists(savedn):
#         try:
#             os.makedirs(savedn)
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
#
#     fig1.savefig(os.path.join(savedn, 'all.png'), dpi=1000, transparent=True)
#     fig1.savefig(os.path.join(savedn, 'all.pdf'), dpi=None, transparent=True)
#
#     fig2.savefig(os.path.join(savedn, 'trajectories_and_histogram.png'), dpi=1000, transparent=True)
#     fig2.savefig(os.path.join(savedn, 'trajectories_and_histogram.pdf'), dpi=None, transparent=True)
#
#     fig3.savefig(os.path.join(savedn, 'velocity.png'), dpi=1000, transparent=True)
#     fig3.savefig(os.path.join(savedn, 'velocity.pdf'), dpi=None, transparent=True)
#
#
#


# # Rotate coordinates

# approximate location of jet outflow (on rotation axis)
new_originx = 1640
new_originy = 940


angle_pi71 = np.pi + 0.2682013644206911 # rad
angle_pi73 = np.pi + -0.11809202841909477 # rad

theta_jet = angle_pi71

# rotation matrix
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
df00 = pd.DataFrame()

pmax = np.max(data['particle'].values)
# pmax = 10000

# init fig
fig10, ax10 = plt.subplots(nrows=1, ncols=1, num='redraw trajectories')
ax10.set_xlabel('axial jet coordinate [m]')
ax10.set_ylabel('transverse jet coordinate [m]')


for i in range(0, pmax):

    # organize by particle
    df02 = data_rot[data_rot['particle'] == i]

    if not df02.empty:

        pathlen.append(len(df02))

        ax10.plot(df02['xp'].values, df02['yp'].values)

        # compute difference
        df03 = df02.groupby('particle').diff()#/df02['frame'].values
        dx_rot = df03['xp']
        dy_rot = df03['yp']
        dframe = df03['frame']

        # velocity components
        u_rot = dx_rot/dframe
        v_rot = dy_rot/dframe
        uvel_rot = pd.DataFrame({'u': u_rot})
        vvel_rot = pd.DataFrame({'v': v_rot})

        # join velocities with temp dataframe, then append to new dataframe
        dfvel_rot = df02.join(uvel_rot)
        dfvel_rot = dfvel_rot.join(vvel_rot)
        df00 = df00.append(dfvel_rot)

# initialize grid
# nx = 1640
# ny = 1232
dn = 10

minx = np.min(df00['xp'].values).astype(int)
maxx = np.max(df00['xp'].values).astype(int)
miny = np.min(df00['yp'].values).astype(int)
maxy = np.max(df00['yp'].values).astype(int)


xgrid = np.arange(minx, maxx, dn)
ygrid = np.arange(miny, maxy, dn)
xxgrid, yygrid = np.meshgrid(xgrid, ygrid)
ugrid = np.zeros(np.shape(xxgrid))
vgrid = np.zeros(np.shape(yygrid))
gridcount = np.zeros(np.shape(xxgrid))


# loop through all entries and append velocity to relevant bin
# !!! need to redefine indices so no negative values are called!!!
for j in range(0, len(df00)):
    if not np.isnan(df00['u'].values[j]):
        Iy = ((df00['yp'].values[j] - miny)/dn).astype(int)
        Ix = ((df00['xp'].values[j] - minx)/dn).astype(int)
        ugrid[Iy-1, Ix-1] += df00['u'].values[j]
    if not np.isnan(df00['v'].values[j]):
        Iy = ((df00['yp'].values[j] - miny)/dn).astype(int)
        Ix = ((df00['xp'].values[j] - minx)/dn).astype(int)
        vgrid[Iy-1, Ix-1] += df00['v'].values[j]
        # assume that if ~isnan u then ~isnanv:
        gridcount[Iy-1, Ix-1] += 1.



umean = np.divide(ugrid.astype(float),gridcount.astype(float))
vmean = np.divide(vgrid.astype(float),gridcount.astype(float))


maxu = np.nanmax(umean[:])
minu = np.nanmin(umean[:])
maxv = np.nanmax(vmean[:])
minv = np.nanmin(vmean[:])
clrmax = np.max([np.abs(maxu), np.abs(minu), np.abs(maxv), np.abs(minv)])


fig11, ax11 = plt.subplots(nrows=2, ncols=1, figsize=(9.3, 7.5))
im0 = ax11[0].imshow(umean, origin='lower', extent=(minx, maxx, miny, maxy), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
im1 = ax11[1].imshow(vmean, origin='lower', extent=(minx, maxx, miny, maxy), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
# fig.colorbar()
# plt.set_cmap('bwr')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im0, cax=cbar_ax)
plt.show()


fig2 = plt.figure(num='trajectory data', figsize=(10,6))
ax22 = plt.subplot(222)
ax22.imshow(umean, origin='lower', extent=(minx, maxx, miny, maxy), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
ax24 = plt.subplot(224)
ax24.imshow(umean, origin='lower', extent=(minx, maxx, miny, maxy), cmap=plt.cm.bwr, vmin=-clrmax, vmax=clrmax)
ax23 = plt.subplot(223)
ax23.hist(pathlen, bins=30)




fig2 = plt.figure(num='trajectory data', figsize=(10,6))
ax23 = plt.subplot(223)
ax23.hist(pathlen, bins=30)
ax23.set_xlabel('path length [pix]')
ax23.set_ylabel('count')
ax23.text(250, 250, 'trajectories=' + str(len(pathlen)))

ax22 = plt.subplot(222)
im22 = ax22.imshow(umean)
# ax22.set_ylabel('y [pix]')
ax22.text(25, 25, 'u component')

# # fig2.tight_layout()
#
# ax24 = plt.subplot(224)
# im24 = ax24.imshow(vmean)
# ax24.set_xlabel('x [pix]')
# ax24.set_ylabel('y [pix]')
# ax24.text(25, 25, 'v component')
#
#
# fig2.tight_layout()
#
# plt.set_cmap('bwr')
#
# fig2.subplots_adjust(right=0.8)
# cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
# clb = fig2.colorbar(im22, cax=cbar_ax)
# # plt.show()
# im22.set_clim(vmin=-6, vmax=6)
# im24.set_clim(vmin=-6, vmax=6)
# clb.ax.set_title('speed [m/s]')
#
# # fig2.tight_layout()
#
#
#
#
# # fig1 = plt.figure(figsize=(12,8), num='trajectories')
# ax21 = plt.subplot(221)
# tp.plot_traj(data, colorby='particle')
# # ax21.xlim([0, 1640])
# # ax21.ylim([1232, 0])
#
#
# fig2 = plt.figure(num='velocity', figsize=(5,6))
#
# ax22 = plt.subplot(211)
# im22 = ax22.imshow(umean)
# ax22.set_ylabel('y [pix]')
# ax22.text(25, 25, 'u component')
#
# ax24 = plt.subplot(212)
# im24 = ax24.imshow(vmean)
# ax24.set_xlabel('x [pix]')
# ax24.set_ylabel('y [pix]')
# ax24.text(25, 25, 'v component')
#
#
# fig2.tight_layout()
#
# plt.set_cmap('bwr')
#
# fig2.subplots_adjust(right=0.8)
# cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
# clb = fig2.colorbar(im22, cax=cbar_ax)
# # plt.show()
# im22.set_clim(vmin=-6, vmax=6)
# im24.set_clim(vmin=-6, vmax=6)
# clb.ax.set_title('speed [pix/s]')
#
# # fig2.tight_layout()
#
#
# fig3 = plt.figure(num='trajectories', figsize=(9,6))
# ax32 = plt.subplot(222)
# ax32.hist(pathlen, bins=30)
# ax32.set_xlabel('path length [pix]')
# ax32.set_ylabel('count')
# ax32.text(250, 250, 'trajectories=' + str(len(pathlen)))
#
# # fig1 = plt.figure(figsize=(12,8), num='trajectories')
# ax31 = plt.subplot(221)
# tp.plot_traj(data, colorby='particle')
# # ax21.xlim([0, 1640])
# # ax21.ylim([1232, 0])
#
# saveFlag = 0
# if saveFlag is 1:
#     savedn = os.path.join('C:\\','Projects','ptv-aquatron2019','reports','figures','trajectories', vid)
#     if not os.path.exists(savedn):
#         try:
#             os.makedirs(savedn)
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
#
#     fig1.savefig(os.path.join(savedn, 'all.png'), dpi=1000, transparent=True)
#     fig1.savefig(os.path.join(savedn, 'all.pdf'), dpi=None, transparent=True)
#
#     fig2.savefig(os.path.join(savedn, 'trajectories_and_histogram.png'), dpi=1000, transparent=True)
#     fig2.savefig(os.path.join(savedn, 'trajectories_and_histogram.pdf'), dpi=None, transparent=True)
#
#     fig3.savefig(os.path.join(savedn, 'velocity.png'), dpi=1000, transparent=True)
#     fig3.savefig(os.path.join(savedn, 'velocity.pdf'), dpi=None, transparent=True)
#
#
#
