import numpy as np
import matplotlib.pylab as plt
import scipy.integrate
import time as time
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import pandas as pd
from matplotlib import rc
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import sys
import numba
from numba import jit
import healpy as hp
import functools
import itertools
import operator
import matplotlib
from matplotlib import rc
import h5py
import os
from os.path import exists
import treecorr
matplotlib.rcParams['axes.linewidth'] = 2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Density threshold:
delta_t = 0.2

# Bin of radius:
Rmin = 3
Rmax = 4

print ("Rmin, Rmax: ", Rmin, Rmax)
# Choose the binning:
rmin = 0.0
rmax = 8 * Rmax

nbins = 80

b = [ [[0, 500], [0, 500], [0, 500]], [[0, 500], [0, 500], [500, 1000]], [[0, 500], [500, 1000], [0, 500]], [[0, 500],[500, 1000],[500, 1000]] ]

# Import simulation:
hf = h5py.File("/datadec/cppm/renanbos/MDPL2_h5/snap_130_all_res_1.hdf5", "r")
print ("Simulation Imported. Now concatenating all subboxes in the same array ... ")

pos_raw = []
for i in range(1919):
    pos_raw.append(hf[list(hf.keys())[i]][()])

pos = np.concatenate(pos_raw)

x = np.array(pos[:, 0])
y = np.array(pos[:, 1])
z = np.array(pos[:, 2])

x = np.array(x.tolist())
y = np.array(y.tolist())
z = np.array(z.tolist())

# Select points in a certain range of zs:
points = np.column_stack((x, y, z))

X_v_all = []
Y_v_all = []
Z_v_all = []
R_v_all = []

for i in range(len(b)):
#    w = (x > b[i][0][0]) & (x < b[i][0][1]) & (y > b[i][1][0]) & (y < b[i][1][1]) & (z > b[i][2][0]) & (z < b[i][2][1])

    voids = np.loadtxt("/datadec/cppm/renanbos/MDPL2_h5/void_catalog_3D_MDPL2_res_1_delta_t_" + np.str(delta_t) + "_xmin_" + np.str(b[i][0][0]) + "_xmax_" + np.str(b[i][0][1]) + "_ymin_" + np.str(b[i][1][0]) + "_ymax_" + np.str(b[i][1][1]) + "_zmin_" + np.str(b[i][2][0]) + "_zmax_" + np.str(b[i][2][1]) + ".dat")

    R_v = voids[:, 3] 
    X_v = voids[:, 0]
    Y_v = voids[:, 1]
    Z_v = voids[:, 2]

    w = (R_v > Rmin) & (R_v < Rmax) & (X_v > b[i][0][0] + 50) & (X_v < b[i][0][1] - 50) & (Y_v > b[i][1][0] + 50) & (Y_v < b[i][1][1] - 50) & (Z_v > b[i][2][0] + 50) & (Z_v < b[i][2][1] - 50) 

    X_v_all.append(X_v[w])
    Y_v_all.append(Y_v[w])
    Z_v_all.append(Z_v[w])
    R_v_all.append(R_v[w])


cat_v = treecorr.Catalog(x = np.hstack(X_v_all), y = np.hstack(Y_v_all), z = np.hstack(Z_v_all))
cat_p = treecorr.Catalog(x = points[:, 0], y = points[:, 1], z = points[:, 2])

vp = treecorr.NNCorrelation(min_sep = 0.1, max_sep = 50, nbins = 50, bin_type = 'Linear')

print ("Counting DD ...")
ti = time.time()
vp.process_cross(cat_v, cat_p, num_threads = 56)
tf = time.time()
print ("Time: ", (tf - ti)/60.)

xi_p = vp.npairs/len(np.hstack(X_v_all))/(4 * np.pi * vp.rnom**2. * (50 - 0.1)/50) - 1

np.savetxt("/datadec/cppm/renanbos/MDPL2_h5/Void_Profile_delta_t" + np.str(delta_t) + "_Rmin_" + np.str(Rmin) + "_Rmax_" + np.str(Rmax) + "_500_box_all.dat", np.column_stack((vp.rnom, xi_p)))

print ("xi: ", xi_p)

