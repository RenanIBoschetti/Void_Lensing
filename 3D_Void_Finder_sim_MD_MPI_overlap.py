import numpy as np
import matplotlib.pylab as plt
import scipy.integrate
import time as time
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import pandas as pd
from matplotlib import rc
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
matplotlib.rcParams['axes.linewidth'] = 2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Define the mask: (Attention: one must change L if the box is not cubic)
x_max = 650
x_min = 450
y_max = 650
y_min = 450
z_max = 650
z_min = 450

# Cat density:
den_total = 1

# Density threshold:
delta_t = 0.2

print ("delta_t ,xmin, xmax, ymin, ymax, zmin, zmax: ", delta_t, x_min, x_max, y_min, y_max, z_min, z_max)

# Import simulation:
hf = h5py.File("/datadec/cppm/renanbos/MDPL2_h5/snap_130_all_res_1.hdf5", "r")
print ("Simulation Imported. Now concatenating all subboxes in the same array ... ")

pos_raw = []
for i in range(1919):
    pos_raw.append(hf[list(hf.keys())[i]][()])

pos = np.concatenate(pos_raw)

w = (pos[:, 0] > x_min) & (pos[:, 0] < x_max) & (pos[:, 1] > y_min) & (pos[:, 1] < y_max) & (pos[:, 2] > z_min) & (pos[:, 2] < z_max) 

x = np.array(pos[:, 0][w])
y = np.array(pos[:, 1][w])
z = np.array(pos[:, 2][w])

#L = x_max - x_min

x = np.array(x.tolist())
y = np.array(y.tolist())
z = np.array(z.tolist())

# Compute centers of circumscripted spheres:
@jit(nopython = True)
def sphere_circumscript(T):
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = T
    A = np.array([ [x1, y1, z1, 1], [x2, y2, z2, 1], [x3, y3, z3, 1], [x4, y4, z4, 1] ])
    B = np.array([ -x1**2.-y1**2.-z1**2., -x2**2.-y2**2.-z2**2., -x3**2.-y3**2.-z3**2.,-x4**2.-y4**2.-z4**2.])
    X = np.linalg.solve(A, B)
    x, y, z = -X[0]/2., -X[1]/2., -X[2]/2.
    r = np.sqrt((X[0]**2. + X[1]**2. + X[2]**2.)/4 - X[3])
    return x, y, z, r

# Select points in a certain range of zs:
points = np.column_stack((x,y, z))

ti = time.time()
print ("Building the Delaunay triangulation ... ")
tri = Delaunay(points)
print ("Done!")
vertices = points[tri.vertices]
tf = time.time()
print ("Time for Delaunay triangulation: ", tf - ti)

centers = np.zeros((len(tri.vertices), 3))
radius = np.zeros(len(tri.vertices))
ti = time.time()
for i in range(len(tri.vertices)):
    centers[i] = sphere_circumscript(vertices[i])[:3]
    radius[i] = sphere_circumscript(vertices[i])[3]
tf = time.time()

# Determine the mean density of galaxies
#total_vol = L**3.  
#den_total = len(points)/total_vol
print ("The density is: ", den_total)

w = np.intersect1d(  np.where(centers[:, 0] >  1 + x_min)[0], np.where(centers[:, 0] < x_max - 1)[0]) 

w_ = np.intersect1d( w, np.where(centers[:, 1] > 1 + y_min)[0]) 

w__ = np.intersect1d( np.where(centers[:, 1] < y_max - 1)[0], w_ )

w___ = np.intersect1d( np.where(centers[:, 2] > 1 + z_min)[0], w__ )

w____ = np.intersect1d( np.where(centers[:, 2] < z_max - 1)[0],w___ )

centers_ = centers[w____]
radius_ = radius[w____]

sorted_radii = np.argsort(radius_)
centers__ = centers_[sorted_radii].reshape(len(centers_[sorted_radii]), 3)
radius__ = radius_[sorted_radii].reshape(len(radius_[sorted_radii]))

np.savetxt("/datadec/cppm/renanbos/MDPL2_h5/centers.dat", centers__)
np.savetxt("/datadec/cppm/renanbos/MDPL2_h5/radii_xmin_" + np.str(x_min) + "_xmax_" + np.str(x_max) + "_ymin_" + np.str(y_min) + "_ymax_" + np.str(y_max) + "_zmin_" + np.str(z_min) + "_zmax_" + np.str(z_max) + ".dat", radius__)

#w_plot = points[:, 2] < 30
#w_centers = centers__[:, 2] < 15
#centers_plot = centers__[w_centers]
#print ("Plotting candidates centers ...")
#plt.figure()
#plt.scatter(points[:, 0][w_plot], points[:, 1][w_plot], s = 0.01, linewidth = 0.05 )
#plt.scatter( centers[len(centers)-1][0] + 0.08 * np.cos(theta), centers[len(centers)-1][1] + 0.08 * np.sin(theta), color = 'red', s = 1)
#plt.scatter( centers__[w_centers][len(centers__) - 10000:,0], centers__[w_centers][len(centers__) - 10000:,1], color = 'red', s = 0.1, linewidth = 0.1)
#plt.scatter( centers__[w_centers][:len(centers__) - 100000,0], centers__[w_centers][:len(centers__) - 100000,1], color = 'green', s = 0.1, linewidth = 0.1)
#plt.xlabel(r"$x[h^{-1}$" + r"$\textrm{Mpc}]$", fontsize = 14)
#plt.ylabel(r"$y[h^{-1}$" + r"$\textrm{Mpc}]$" , fontsize = 14)
#plt.savefig("/datadec/cppm/renanbos/MDPL2_h5/test_centers_3D_MDPL2.png", dpi = 1000)


import itertools
import operator
# The new, simplest version:
candidates = []
radius_candidates = []
centers_candidates = []
ti = time.time
N_candidates = len(centers__)
# Define density threshold:

ball = 40

file = open("/datadec/cppm/renanbos/MDPL2_h5/void_catalog_3D_MDPL2_res_1_delta_t_" + np.str(delta_t) + "_xmin_" + np.str(x_min) + "_xmax_" + np.str(x_max) + "_ymin_" + np.str(y_min) + "_ymax_" + np.str(y_max) + "_zmin_" + np.str(z_min) + "_zmax_" + np.str(z_max) + "_all_cand.dat", 'w')
#for j in range(N_candidates):
def Grow_candidates(i):
    ti = time.time()

    pos = len(centers__) - (i + 1) # Take the ith largest area as the center of void candidate
#    pos = i
    center = centers__[pos] # take the center of the current candidate
#    print (i, center, radius__[pos])

    # Select points near to the current center
    w = (np.sqrt( (points[:, 0] - centers__[pos][0])**2. + (points[:,1] - centers__[pos][1])**2. + (points[:, 2] - centers__[pos][2])**2.) <  ball ) 
    points_near = points[w]
    
    dists = np.sqrt( (center[0] - points_near[:, 0])**2. + (center[1] - points_near[:, 1])**2. + (center[2] - points_near[:, 2])**2. )

#    radii = np.sort( dists )
    sort_ind = dists.argsort()
    radii = dists[sort_ind]
    n_p = np.arange(1, len(radii) + 1, 1)
    den = n_p/((4./3.) * np.pi * radii**3. )

    if not (den > delta_t * den_total).any():
        return 0, 0, 0, 0

    ind = np.where(den > delta_t * den_total)[0][0]
    radius = radii[ind]
    points_near_sorted = points_near[sort_ind]
#    void_candidate = points_near_sorted[:ind]
    
    theta1 = np.linspace(0., 3.1415926, 5)
    phi = np.linspace(0., 2 * 3.14, 5)
    theta2 = np.linspace(0., 3.1415926, 25)
    ang1 = np.reshape(np.outer(np.cos(phi), np.sin(theta1)), 25)
    ang2 = np.reshape(np.outer(np.sin(phi), np.sin(theta1)), 25)
    x_v = center[0] + radius * ang1
    y_v = center[1] + radius * ang2
    z_v = center[2] + radius * np.cos(theta2)

    tf = time.time()
    if i == 0 or i ==10 or i == 1000 or i == 10000:
        print ("Time for the first candidate: ", tf - ti)
        print ("Estimated time for all: ", (tf - ti) * np.int(N_candidates * 0.01)/3600., "h" )
    
    if any(x_v > x_max) or any(y_v > y_max) or any(z_v > z_max) or any(x_v < x_min) or any(y_v < y_min) or any(z_v < z_min):
        
        return 0, 0, 0, 0
    else:
        file.write(np.str(center[0]) + " " + np.str(center[1]) + " " + np.str(center[2]) + " " + np.str(radius) + "\n")
        if i < 20:
            print ("minimum distance: ", np.min(dists))
            print ("i, radius, center: ", i,radius__[pos], center)
        return center[0], center[1], center[2], radius
print ("We are going to use ", np.int(N_candidates * 0.01), " candidates.")

context = multiprocessing.get_context('fork')
#pool = context.Pool(processes= multiprocessing.cpu_count())
pool = context.Pool(processes = multiprocessing.cpu_count(), maxtasksperchild = 100)
#print ( "Growing candidates from 272786 to 409179 ...")
ti = time.time()
candidates = pool.map(Grow_candidates, np.arange(np.int(N_candidates * 0.01)))
tf = time.time()
time_mpi = tf - ti
pool.close()
file.close()
print ("Candidates found! It took ", time_mpi/3600.,"h")
w = (np.array(candidates)[:, 0] != 0) & (np.array(candidates)[:, 1] != 0) & (np.array(candidates)[:, 2] != 0) & (np.array(candidates)[:, 3] != 0)
can = np.array(candidates)[w]

can_sorted = can[np.argsort(can[:, 3])[::-1]]

print ("Cleaning the catalogue ...")
voids = []
for i in range(len(can_sorted)):
    if i == 0:
        voids.append(can_sorted[0])

    if i == 1:
        distance = np.sqrt( (voids[0][0] - can_sorted[i][0])**2. + (voids[0][1] - can_sorted[i][1])**2. + (voids[0][2] - can_sorted[i][2])**2.)
        radius_distance = can_sorted[i][3] + voids[0][3]
        if radius_distance < distance:
            voids.append(can_sorted[1])
    if i > 1:
        distance = np.sqrt( (np.array(voids)[:, 0] - can_sorted[i][0])**2. + (np.array(voids)[:, 1] - can_sorted[i][1])**2. + (np.array(voids)[:, 2] - can_sorted[i][2])**2.)
        radius_distance = can_sorted[i][3] + np.array(voids)[:, 3]
        if not any(radius_distance > distance):
            voids.append(can_sorted[i])
voids_catalog = np.array(voids)

np.savetxt("/datadec/cppm/renanbos/MDPL2_h5/void_catalog_3D_MDPL2_res_1_delta_t_" + np.str(delta_t) + "_xmin_" + np.str(x_min) + "_xmax_" + np.str(x_max) + "_ymin_" + np.str(y_min) + "_ymax_" + np.str(y_max) + "_zmin_" + np.str(z_min) + "_zmax_" + np.str(z_max) + ".dat", voids_catalog)


