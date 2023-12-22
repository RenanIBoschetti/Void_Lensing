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
import os
import h5py
matplotlib.rcParams['axes.linewidth'] = 2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Define mask:
x_max = 100
x_min = 0
y_max = 500
y_min = 0
z_max = 500
z_min = 0

print ("xmin, xmax, ymin, ymax, z_min, z_max: ", x_min, x_max, y_min, y_max, z_min, z_max)

# Density threshold:
delta_t = 0.5

print("delta_t: ", delta_t)

# 3D density:
d3_density = 1.3

# Fraction of candidates to use:
f_can = 0.1

hf = h5py.File("/datadec/cppm/renanbos/uchuu/uchuu_045_res_1.hdf5", "r")

pos_raw = []
for i in range(200):
    pos_raw.append(hf[list(hf.keys())[i]][()])

pos = np.concatenate(pos_raw)

print ("pos min max x: ", np.min(pos[:, 0]), np.max(pos[:, 0]) )
print ("pos min max y: ", np.min(pos[:, 1]), np.max(pos[:, 1]) )
print ("pos min max z: ", np.min(pos[:, 2]), np.max(pos[:, 2]) )

#w_xy = (pos[:, 0] > x_min) & (pos[:, 0] < x_max) & (pos[:, 1] > y_min) & (pos[:, 1] < y_max) 

#w_xz = (pos[:, 0] > x_min) & (pos[:, 0] < x_max) & (pos[:, 2] > z_min) & (pos[:, 2] < z_max) 

w_yz = (pos[:, 1] > y_min) & (pos[:, 1] < y_max) & (pos[:, 2] > z_min) & (pos[:, 2] < z_max) 

x = np.array(pos[:, 0][w_yz])
y = np.array(pos[:, 1][w_yz])
z = np.array(pos[:, 2][w_yz])

print ("x, y, z: ", x, y, z)

x = np.array(x.tolist())
y = np.array(y.tolist())
z = np.array(z.tolist())

print ("x, y, z new: ", x, y, z)
# Alternative way of computing centers of circumscripted circles:
@jit(nopython = True)
def circle_circumscript(T):
    (x1, y1), (x2, y2), (x3, y3) = T
    A = np.array([ [x1, y1, 1], [x2, y2, 1], [x3, y3, 1] ])
    B = np.array([ -x1**2.-y1**2., -x2**2.-y2**2, -x3**2.-y3**2 ])
    X = np.linalg.solve(A, B)
    x, y = -X[0]/2., -X[1]/2.
    r = np.sqrt((X[0]**2. + X[1]**2.)/4-X[2])
    return x, y, r

# Select points in a certain range of zs:
points_ = np.column_stack((y, z))

L = y_max - y_min
w = (x > x_min) & (x < x_max)
points = points_[w]

X = points[:, 0]
Y = points[:, 1]

print ("X_max, Y_max", np.max(X), np.max(Y))

print ("X, Y: ", X, Y)


print ("Building the Delaunay triangulation ... ")
tri = Delaunay(points)
print ("Done!")
vertices = points[tri.vertices]

centers = np.zeros((len(tri.vertices), 2))
radius = np.zeros(len(tri.vertices))
ti = time.time()
for i in range(len(tri.vertices)):
    centers[i] = circle_circumscript(vertices[i])[:2]
    radius[i] = circle_circumscript(vertices[i])[2]
tf = time.time()

# Determine the mean density of galaxies
total_area = L**2.  
den_total = d3_density * (x_max - x_min)

print ("The density is: ", den_total)

w = np.intersect1d(  np.where(centers[:, 0] > y_min)[0], np.where(centers[:, 0] < y_max)[0]) 

w_ = np.intersect1d( w, np.where(centers[:, 1] > z_min )[0]) 

w__ = np.intersect1d( np.where(centers[:, 1] < z_max )[0], w_ )

centers_ = centers[w__]
radius_ = radius[w__]

sorted_radii = np.argsort(radius_)
centers__ = centers_[sorted_radii].reshape(len(centers_[sorted_radii]), 2)
radius__ = radius_[sorted_radii].reshape(len(radius_[sorted_radii]))

print ("Plotting candidates centers ...")
plt.figure()
plt.scatter(points[:, 0], points[:, 1], s = 0.01, linewidth = 0.05 )
#plt.scatter( centers[len(centers)-1][0] + 0.08 * np.cos(theta), centers[len(centers)-1][1] + 0.08 * np.sin(theta), color = 'red', s = 1)
plt.scatter( centers__[(len(centers__) - 100000):][:,0], centers__[(len(centers__) - 100000):][:,1], color = 'red', s = 0.05)
plt.xlabel(r"$x[h^{-1}$" + r"$\textrm{Mpc}]$", fontsize = 14)
plt.ylabel(r"$y[h^{-1}$" + r"$\textrm{Mpc}]$" , fontsize = 14)
plt.savefig("/datadec/cppm/renanbos/uchuu/test_centers_uchuu.png", dpi = 1000)

import itertools
import operator
# The new, simplest version:
candidates = []
radius_candidates = []
centers_candidates = []
ti = time.time
N_candidates = len(centers)
# Define density threshold:

ball = 50

file = open("/datadec/cppm/renanbos/uchuu/void_catalog_yz_uchuu_res_1_f_can_" + str(f_can) + "_delta_t_" + str(delta_t) + "_xmin_" + str(x_min) + "_xmax_" + str(x_max) + "_ymin_"  + str(y_min) + "_ymax_" + str(y_max) + "_zmin_" + str(z_min) + "_zmax_"+ str(z_max) + "_all_cand.dat", 'w')

#for j in range(N_candidates):
def Grow_candidates(i):
    ti = time.time()


    pos = len(centers__) - (i + 1) # Take the ith largest area as the center of void candidate
#    pos = i
    center = centers__[pos] # take the center of the current candidate
#    print (i, center, radius__[pos])
    if i == 0:
        print ("First radius: ", radius__[pos], center)
    # Select points near to the current center
    w = (np.sqrt( (points[:, 0] - centers__[pos][0])**2. + (points[:,1] - centers__[pos][1])**2.) <  ball ) 
    points_near = points[w]
    
    dists = np.sqrt( (center[0] - points_near[:, 0])**2. + (center[1] - points_near[:, 1])**2.)
    radii = np.sort( dists )
    sort_ind = dists.argsort()
    n_p = np.arange(1, len(radii) + 1, 1)
    den = n_p/( np.pi * radii**2. )

    if not (den > delta_t * den_total).any():
        return 0, 0, 0

    ind = np.where(den > delta_t * den_total)[0][0]
    radius = radii[ind]
#    points_near_sorted = points_near[sort_ind]
#    void_candidate = points_near_sorted[:ind]
    
    theta = np.linspace(0., 2 * np.pi, 10)
    x_v = center[0] + radius * np.cos(theta)
    y_v = center[1] + radius * np.sin(theta)
    tf = time.time()
    if i == 0:
        print ("Time for the first candidate: ", tf - ti)
    
    if any(x_v > y_max) or any(x_v < y_min) or any(y_v > z_max) or any(y_v < z_min):
        return 0, 0, 0
    else:
        file.write(str(center[0]) + " " + str(center[1]) + " " + str(radius) + "\n")
        return center[0], center[1], radius

print ("We are going to use ", int(N_candidates * f_can), " candidates.")

context = multiprocessing.get_context('fork')
#pool = context.Pool(processes= multiprocessing.cpu_count())
pool = context.Pool(processes = multiprocessing.cpu_count(), maxtasksperchild = 100)
print ( "Growing candidates ...")
ti = time.time()
candidates = pool.map(Grow_candidates, np.arange(np.int(N_candidates * f_can)))
tf = time.time()
time_mpi = tf - ti
pool.close()
file.close()

w = (np.array(candidates)[:, 0] != 0) & (np.array(candidates)[:, 1] != 0) & (np.array(candidates)[:, 2] != 0)
can = np.array(candidates)[w]

can_sorted = can[np.argsort(can[:, 2])[::-1]]


print ("Cleaning the catalogue ...")
voids = []
for i in range(len(can_sorted[:np.where(can_sorted[:, 2] < 1)[0][0]])):
    if i == 0:
        voids.append(can_sorted[0])

    if i == 1:
        distance = np.sqrt( (voids[0][0] - can_sorted[i][0])**2. + (voids[0][1] - can_sorted[i][1])**2.)
        radius_distance = can_sorted[i][2] + voids[0][2]
        if radius_distance < distance:
            voids.append(can_sorted[1])
    if i > 1:
        distance = np.sqrt( (np.array(voids)[:, 0] - can_sorted[i][0])**2. + (np.array(voids)[:, 1] - can_sorted[i][1])**2.)
        radius_distance = can_sorted[i][2] + np.array(voids)[:, 2]
        if not any(radius_distance > distance):
            voids.append(can_sorted[i])


voids_catalog = np.array(voids)

np.savetxt("/datadec/cppm/renanbos/uchuu/void_catalog_yz_uchuu_res_1_f_can_" + str(f_can) + "_delta_t_" + str(delta_t) + "_xmin_" + str(x_min) + "_xmax_" + str(x_max) + "_ymin_"  + str(y_min) + "_ymax_" + str(y_max) + "_zmin_" + str(z_min) + "_zmax_"+ str(z_max) + ".dat", voids_catalog)

theta = np.arange(0., 6.28, 0.01)


plt.figure()
plt.scatter(points[:,0], points[:,1], s = 0.01, linewidth = 0.01)
#plt.scatter(rand_gals['dec'][rindex][index_sphere], rand_gals['ra'][rand_index][index_sphere], s = 0.01, linewidth = 0.05)
for i in range(len(voids_catalog)):
    r = voids_catalog[i, 2]
    plt.plot(r * np.cos(theta)+voids_catalog[i, 0], r * np.sin(theta)+voids_catalog[i, 1], color = 'black', linewidth = 0.2  )        

    #   plt.scatter( centers[len(centers)-2700:][:,0], centers[len(centers)-2700:][:, 1], color = 'red', s = 1)
 #   plt.scatter( centers[len(centers)-1337][0], centers[len(centers)-1337][1], color = 'green', s = 1)


plt.xlabel(r"$x[h^{-1}$" + r"$\textrm{Mpc}]$", fontsize = 14)
plt.ylabel(r"$y[h^{-1}$" + r"$\textrm{Mpc}]$" , fontsize = 14)
#plt.savefig("/Users/renanbos/Desktop/Important_r esults/new_alg_stage1mock_delta_t_" + np.str(delta_t) + "_zmax_" +np.str(z_max) + "_zmin_" + np.str(z_min) + "_new_cr_dxi" + np.str(dXi) + ".pdf")
#plt.scatter(center_dec, center_ra, s = 1, linewidth = 1, color = 'red')
#plt.xlim(2.10,2.25)
#plt.ylim(-0.1, -0.05)
#plt.savefig("/Users/renanbos/Desktop/Void_Finder/voids_found_excluding_smaller_threshold_"+np.str(delta_t)+"_triangle_area_"+np.str(np.round(total_area,2))+"_cutting_small.pdf")
#plt.savefig("/Users/renanbos/rand_voids_found_excluding_smaller_"+np.str(delta_t)+"_triangle_area_"+np.str(np.round(total_area,2))+"_"+np.str(percentage)+"z_range"+np.str(z_min)+"_"+np.str(z_max)+"_scatter_version_new_intersection.pdf")
plt.savefig("/datadec/cppm/renanbos/uchuu/voids_yz_uchuu_res_1_f_can_" + str(f_can) + "_delta_t_" + str(delta_t) + "_xmin_" + str(x_min) + "_xmax_" + str(x_max) + "_ymin_"  + str(y_min) + "_ymax_" + str(y_max) + "_zmin_" + str(z_min) + "_zmax_"+ str(z_max) + ".png", dpi = 1500)



