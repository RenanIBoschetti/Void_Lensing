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
import pandas 
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
from astropy.table import Table
import math
import pyarrow.parquet as pq
from sklearn.utils import shuffle
from scipy import interpolate
from astropy.io import fits
from astropy.table import Table
from matplotlib import rc
import pandas as pd
import matplotlib.pylab as plt
matplotlib.rcParams['axes.linewidth'] = 2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Cosmology:
Om = 0.286
Ol = 1 - Om
H0 = 67.77
h = 0.677

# Set resolution:
res_set = 0.01

# Calculate the comoving distances:
def H(z):
    return H0*np.sqrt(Om*(1+z)**3.+Ol)

z_vec = np.linspace(0.01, 1.7, 2000)
r_c = np.ones(len(z_vec))

# Angular diameter distance function: ATENTION: this is in Mpc/h 
for i in range(len(z_vec)): 
    r_c[i] = 2997.92*scipy.integrate.simps(1./np.sqrt( Om*( 1.+np.linspace(0., z_vec[i], 1000) )**3.+Ol ) , np.linspace(0., z_vec[i], 1000)) 

r_c_interp = interpolate.interp1d(z_vec, r_c)
z_interp = interpolate.interp1d(r_c, z_vec)

# Density threshold:
delta_t = 0.5

# Which tracer:
tracer = 'BGS'


# Bin of redshift:
zmin = 0.1
zmax = 
zmean = zmin + (zmax - zmin)/2.

# Bin of radius:
Rmin = 10
Rmax = 100

print ("Rmin, Rmax: ", Rmin, Rmax)

# Choose the binning:
nbins_sigma = 30   
rmin_sigma = 0.0
rmax_sigma = 10         

# Import box:
print ("Importing galaxies ...")
# Choose minimum value of redshift (maximum value is fixed by dXi)
z_min = zmin

# The size of projected bin in units of Mpc/h
dXi = 10

# Cosmology:
Om = 0.286
Ol = 1 - Om
H0 = 67.77
h = 0.677

# Define the directory where the files are located
dir_path = "/datadec/desi/c3wg/buzzard-0/particles/downsampled_dm_norot/"
n_files = 64

print ("Importing particles ...")
catalogs = []
for i in range(n_files):
    data = Table.read('/datadec/desi/c3wg/buzzard-0/particles/downsampled_dm_norot/downsampled_particles.' + str(i) + '.fits', format='fits')['PX', 'PY', 'PZ', 'Z_COS']
    catalog = data.to_pandas()
    catalogs.append(catalog)

for i in range(len(catalogs)):
    if i == 0:
        data_part_aux = catalogs[i]
    else:    
        data_part_aux = np.concatenate((data_part_aux, catalogs[i]))
    part =  pd.DataFrame(data_part_aux, columns = ['PX', 'PY', 'PZ', 'Z_COS'])
print ("Particles: ", part)

w = (part['Z_COS'] > zmin) & (part['Z_COS'] < zmax) & (part['PY'] > 0) & (part['PZ'] > 0)

x = part['PX'][w]/np.sqrt( part['PX'][w]**2. + part['PY'][w]**2. + part['PZ'][w]**2.  )
y = part['PY'][w]/np.sqrt( part['PX'][w]**2. + part['PY'][w]**2. + part['PZ'][w]**2.  )
z = part['PZ'][w]/np.sqrt( part['PX'][w]**2. + part['PY'][w]**2. + part['PZ'][w]**2.  )

part_ra = np.arctan2(y, x)
part_dec = np.arcsin(z)

ra_max = np.max(part_ra)
ra_min = np.min(part_ra)
dec_max = np.max(part_dec)
dec_min = np.min(part_dec)

total_area = np.abs(ra_max - ra_min) * np.abs(np.cos(dec_min) - np.cos(dec_max))
print ("total_area: ", total_area)
#total_area = 4 * np.pi / 4.
#print ("total_area: ", total_area)
den_total = len(x)/total_area

print ("Number of particles: ", len(x) )
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(x, y, z, s = 0.05, linewidth = 0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
ax.view_init(elev = 20., azim = 80)
plt.savefig("/datadec/cppm/renanbos/test_buzz_part.png", dpi = 900 )

# Import voids:
cat = np.loadtxt('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + str(np.round(zmin, 3)) + "_zmax_" + str(np.round(zmax, 3)) + "_last.dat")

print ("cat: ", cat)
w = (cat[:, 3] * r_c_interp(zmean) > Rmin) & (cat[:, 3] * r_c_interp(zmean) < Rmax)

cat = cat[w]

lenses_ra = np.zeros(len(cat))
lenses_dec = np.zeros(len(cat))
for i in range(len(lenses_ra)):
    lenses_ra[i] = math.atan2(cat[i, 1], cat[i, 0])
    lenses_dec[i] = np.arcsin(cat[i, 2])


w_2 = (lenses_ra < np.max(part_ra) ) & (lenses_ra > np.min(part_ra) ) & (lenses_dec > np.min(part_dec) )

cat = cat[w_2]
lenses_ra = lenses_ra[w_2]
lenses_dec = lenses_dec[w_2]

print ("We are going to use ", len(cat), " voids.")

def dist_void_gal( ra_l, ra_s, dec_l, dec_s ):
    x_l = np.cos(dec_l) * np.cos(ra_l)
    y_l = np.cos(dec_l) * np.sin(ra_l)
    z_l = np.sin(dec_l)
    x_s = np.cos(dec_s) * np.cos(ra_s)
    y_s = np.cos(dec_s) * np.sin(ra_s)
    z_s = np.sin(dec_s)
#def dist_void_gal( x_l, y_l, z_l, x_s, y_s, z_s ):
    return np.arccos(np.dot( np.column_stack( (x_s, y_s, z_s) ), np.array([x_l, y_l, z_l]) ))


def count(dist, nbins,  rmin, rmax ):
    counts, r_rv = np.histogram( dist, bins = nbins, range = [rmin, rmax] )
    return counts, r_rv

def get_vg_corr_2D(j):
    len_ra = lenses_ra[j]
    len_dec = lenses_dec[j]
    len_radius = cat[j, 3]

    sigma = dist_void_gal(len_ra, part_ra, len_dec, part_dec)
#    sigma = dist_void_gal(cat[j, 0], cat[j, 1], cat[j, 2], x, y, z)

    counts, r_rv_sigma = count(sigma/len_radius,  nbins_sigma, rmin_sigma, rmax_sigma)

    dr_rv_sigma = r_rv_sigma[1] - r_rv_sigma[0]
    r_rv_m_sigma = r_rv_sigma[:len(r_rv_sigma)-1] + dr_rv_sigma/2.

    den = counts/( 2 * np.pi * np.sin(r_rv_m_sigma * len_radius) * dr_rv_sigma * len_radius * den_total) - 1

    print ("den: ", den)
    print ("counts: ", counts)
    return den, r_rv_m_sigma


print ("Running the vg ...")
context = multiprocessing.get_context('fork')
#pool = context.Pool(processes= multiprocessing.cpu_count())
pool = context.Pool(processes = multiprocessing.cpu_count(), maxtasksperchild = 100)
vg_2d = pool.map(get_vg_corr_2D, np.arange(len(cat)))
pool.close()

r_rv_m_sigma = vg_2d[0][1]

n_g_2d = np.zeros((len(r_rv_m_sigma)))

for j in range(len(vg_2d)):
    n_g_2d += vg_2d[j][0]/len(cat)

print ("xi: ", n_g_2d)
np.savetxt("/datadec/cppm/renanbos/TwoD_Void_Profile_buzz_dm_lc_gal_field_dXi_" + str(dXi) + "_delta_t" + str(delta_t) + "_Rmin_" + str(Rmin) + "_Rmax_" + str(Rmax) + "_zmin_" + str(zmin) + "_zmax_" + str(zmax) + ".dat", np.column_stack((r_rv_m_sigma, n_g_2d)))

