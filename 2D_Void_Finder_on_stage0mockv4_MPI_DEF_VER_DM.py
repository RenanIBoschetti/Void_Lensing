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
from scipy import interpolate
import numba
from numba import jit
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d
import math
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


######### INPUTS ##########
# Choose between BGS and LGR:
tracer = 'BGS'

# Density threshold in the galaxy field:
delta_t_g = 0.5

# Import b(z):
#bz = np.loadtxt("/datadec/cppm/renanbos/bz.txt")

# Choose minimum value of redshift (maximum value is fixed by dXi)
z_min =  0.1
z_max = 0.3

# The size of projected bin in units of Mpc/h
dXi = 20

# Cosmology:
Om = 0.286
Ol = 1 - Om
H0 = 70
h = H0/100.

# Choose the mask edges:
ra_min = 0
ra_max = np.pi
dec_min = 0
dec_max = np.pi/2.

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

# Transform to radians:
#gals['ra'] = np.radians(gals['ra'])
#gals['dec'] = np.radians(gals['dec'])

# Calculate the comoving distances:
def H(z):
    return H0*np.sqrt(Om*(1+z)**3.+Ol)

z_vec = np.linspace(0.01, 1.7, 2000)
r_c = np.ones(len(z_vec))

# Angular diameter distance function: ATENTION: this is in pc/h (Thats why the factor of 10^6 in front of)
for i in range(len(z_vec)): 
    r_c[i] = 2997.92*scipy.integrate.simps(1./np.sqrt( Om*( 1.+np.linspace(0., z_vec[i], 1000) )**3.+Ol ) , np.linspace(0., z_vec[i], 1000)) 

r_c_interp = interpolate.interp1d(z_vec, r_c)
z_interp = interpolate.interp1d(r_c, z_vec)

zmin = z_min

print ("We are going to find voids in slices of ", dXi, "Mpc/h")
b_c = 0
while z_interp( r_c_interp(zmin) + dXi ) <= z_max:
 
    zmax = z_interp( r_c_interp(zmin) + dXi )
    print ("Finding voids in the catalogue with zmin and zmax: ", zmin, zmax)
    zmean = zmin + (zmax - zmin)/2.
    
    print ("zmax: ", zmax)

    #w = (np.array(gals['ra']) > ra_min) & (np.array(gals['ra']) < ra_max) & (np.array(gals['dec']) > dec_min) & (np.array(gals['dec']) < dec_max)  & (np.array(gals['z'] < z_max)) & (np.array(gals['z'] > z_min))

#    w = (np.array(gals['ra']) > ra_min) & (np.array(gals['ra']) < ra_max) & (np.array(gals['z'] < zmax)) & (np.array(gals['z'] > zmin))

#    gals_ = gals[w]
    
    w = (part['Z_COS'] > zmin) & (part['Z_COS'] < zmax)

    part_ = part[w] 

    ti_tot = time.time()

    # Project all points into a spherical shell:

    points_cart_x = part_['PX']/np.sqrt(part_['PX']**2. + part_['PY']**2. + part_['PZ']**2.)
    points_cart_y = part_['PY']/np.sqrt(part_['PX']**2. + part_['PY']**2. + part_['PZ']**2.)
    points_cart_z = part_['PZ']/np.sqrt(part_['PX']**2. + part_['PY']**2. + part_['PZ']**2.)

    points_cart = np.column_stack( (points_cart_x, points_cart_y, points_cart_z) )

    # Complete with random points:
    def sample_spherical_(npoints, phi_min, phi_max, theta_min, theta_max):
        phi = np.random.uniform(phi_min, phi_max, npoints)
    #    print (np.sort(phi))
        theta = np.random.uniform( theta_min, theta_max, npoints )
    #    print (np.sort(theta))
    #    print (np.sort(np.cos(theta)))
    #    print (np.sort(np.sin(phi)))
        v_n = np.column_stack( (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)) )
        return v_n

    randomns_1 = sample_spherical_(5000,   np.pi,  2 * np.pi, 0,   np.pi/2.)
    randomns_2 = sample_spherical_(5000,   np.pi,  2 * np.pi, 3 *np.pi/2.,   2 * np.pi)
    randomns_3 = sample_spherical_(5000,  0,  np.pi, 3 * np.pi/2., 2 * np.pi )

    randomns_ = np.vstack( (randomns_1, randomns_2, randomns_3) )

    points = np.vstack((points_cart, randomns_))

    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(1e-06)

    excl = []
    for i in pairs:
        excl.append(i[1])

    points = np.delete(points, excl, axis = 0)

    # Perform Voronoi tesselation:
    print ("Performing the Voronoi Tesselation ...")
    vor = SphericalVoronoi(points, radius = 1, center = np.array([0, 0, 0]))

    # Select Voronoi vertices and points that are inside the footprint:
    # Select Voronoi vertices and points that are inside the footprint:
    
    ra_vor = []
    dec_vor = []
    for i in range(len(vor.vertices)):
        ra_vor.append(math.atan2(vor.vertices[i, 1], vor.vertices[i, 0]))
        dec_vor.append(np.arcsin(vor.vertices[i, 2]))
    ra_vor = np.array(ra_vor)
    dec_vor = np.array(dec_vor)

    w = (vor.vertices[:, 1] > 0) & (dec_vor > 0) & (dec_vor < np.pi/2.) 

    vor_vertices = vor.vertices[w]

    # Compute the nearest neighbour between each Voronoi vertice and each galaxy:
    # Array to store the center and radii of circles:

    X = np.vstack( (vor_vertices, points_cart) )
    tree = KDTree(X)

    dist, ind = tree.query(X[:len(vor_vertices)], k = 2)

    circles = np.column_stack( (vor_vertices[:, 0], vor_vertices[:, 1], vor_vertices[:, 2], dist[:, 1]) )

    arg_sort = np.argsort(circles[:, 3])
    circles_sorted = circles[arg_sort][::-1]

    centers = circles_sorted[:, :3]

    centers_ra = []
    centers_dec = []
    for i in range(len(centers)):
        centers_ra.append(math.atan2(centers[i, 1], centers[i, 0]))
        centers_dec.append(np.arcsin(centers[i, 2]))
    centers_ra = np.array(centers_ra)
    centers_dec = np.array(centers_dec)

    total_area = (ra_max - ra_min) * (np.cos(dec_min) - np.cos(dec_max))
    den_total = len(points)/total_area
    N_candidates = len(circles)

    print ("We are going to find voids using ", len(circles), " centers.")

    if zmin < 0.2:

        delta_t = (delta_t_g - 1)/1.37 + 1
    else:
        delta_t = (delta_t_g - 1)/1.48 + 1

    print ("The density threshold is: ", delta_t)
    
#    b_c = b_c + 1
    print (" The voids will grow until they reach the density of ", delta_t * den_total)

    #for j in range(N_candidates):
    def Grow_candidates(i):
        ti = time.time()

    #    pos = len(centers) - (i + 1) # Take the ith largest area as the center of void candidate
        pos = i
        center = centers[pos] # take the center of the current candidate
        center_ra = centers_ra[pos]
        center_dec = centers_dec[pos]
        
        # Select points near to the current center
        w = (np.arccos( points_cart[:, 0] * centers[pos][0] + points_cart[:,1] * centers[pos][1] + points_cart[:,2] * centers[pos][2])  <  0.3 ) 
        points_near = points_cart[w]
        
        dists = np.arccos( points_near[:, 0] * centers[pos][0] + points_near[:,1] * centers[pos][1] + points_near[:,2] * centers[pos][2])
        radii = np.sort( dists )
        sort_ind = dists.argsort()
        n_p = np.arange(1, len(radii) + 1, 1)
        den = n_p/( np.pi * radii**2. )
        
        if not (den > delta_t * den_total).any():
            return 0, 0, 0

        ind = np.where(den > delta_t * den_total)[0][0]
        radius = radii[ind]
        points_near_sorted = points_near[sort_ind]
        void_candidate = points_near_sorted[:ind]
        
        theta = np.linspace(0., 2 * np.pi, 10)
        x = center_ra + radius * np.cos(theta)
        y = center_dec + radius * np.sin(theta)
        tf = time.time()
        if i == 0 or i ==10 or i == 10000:
            print ("Void radius in rad: ", radius)

            print ("Time for one void: ", tf - ti)
            print ("Estimated time for all voids: ", (tf - ti) * int(N_candidates * 0.05)/3600./56, "h")

        if any(x < ra_min) or any(x > ra_max) or any(y < dec_min) or any(y > dec_max):
            return 0, 0, 0
        else:
            return center_ra, center_dec, radius

    context = multiprocessing.get_context('fork')
    #pool = context.Pool(processes= multiprocessing.cpu_count())
    pool = context.Pool(processes = multiprocessing.cpu_count(), maxtasksperchild = 100)
    ti = time.time()
    candidates = pool.map(Grow_candidates, np.arange(int(N_candidates * 0.05)))
    tf = time.time()
    time_mpi = tf - ti
    print ("Time for finding voids: ", tf - ti)
    pool.close()


    w = (np.array(candidates)[:, 0] != 0) & (np.array(candidates)[:, 1] != 0) & (np.array(candidates)[:, 2] != 0)
    can = np.array(candidates)[w]

    can_x = np.cos(can[:, 1]) * np.cos(can[:, 0])
    can_y = np.cos(can[:, 1]) * np.sin(can[:, 0])
    can_z = np.sin(can[:, 1])
    can_cart = np.column_stack( (can_x, can_y, can_z, can[:, 2]) )

    can_cart = can_cart[np.argsort(can_cart[:, 3])[::-1]]

    voids = []

    print ("Cleaning the catalogue ...")
    for i in range(len(can_cart)):
        if i == 0:
            voids.append(can_cart[0])

        if i == 1:
            distance = np.arccos( voids[0][0] * can_cart[i][0] + voids[0][1] * can_cart[i][1] + voids[0][2] * can_cart[i][2] )
            radius_distance = can_cart[i][3] + voids[0][3]
            if radius_distance < distance:
                voids.append(can_cart[1])
        if i > 1:
            distance = np.arccos( np.array(voids)[:, 0] * can_cart[i][0] + np.array(voids)[:, 1] * can_cart[i][1] + np.array(voids)[:, 2] * can_cart[i][2] )
            radius_distance = can_cart[i][3] + np.array(voids)[:, 3]
            if not any(radius_distance > distance):
                voids.append(can_cart[i])

    tf_tot = time.time()

    print ("Total time for finding voids: ", (tf_tot - ti_tot)/60., " min")

    voids_catalog = np.array(voids)

    x_v = voids_catalog[:, 0]
    y_v = voids_catalog[:, 1]
    z_v = voids_catalog[:, 2]

    print ("Saving and plotting results ...")
    np.savetxt("/renoir/renanbos/stage0v4_void_catalogues/2D_catalog_stage0v4_DM_dXi_" + str(dXi)+ "_delta_t_g_" + str(delta_t_g)  + "_zmin_" + str(np.round(zmin,3)) + "_zmax_" + str(np.round(zmax,3)) + "_last_corr_.dat", voids_catalog)

    theta = np.arange(0., 6.28, 0.01)
    ori = [ [20, 80] , [20, 150], [20,30], [60, 80] ]
    for o in range(len(ori)):
        # Plot centers and galaxies:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(points_cart[:, 0], points_cart[:, 1], points_cart[:, 2], s = 0.03, linewidth = 0.02)
        for i in range(len(x_v)):
            r = np.array(voids_catalog[i, 3]) 
            u_ra = math.atan2(y_v[i], x_v[i]) + r
            u_dec = np.arcsin(z_v[i]) 
            ux = np.cos(u_dec) * np.cos(u_ra)
            uy = np.cos(u_dec) * np.sin(u_ra)
            uz = np.sin(u_dec)
            etax = - x_v[i] + ux 
            etay = - y_v[i] + uy
            etaz = - z_v[i] + uz
            eta = np.array([etax, etay, etaz])/np.sqrt(etax**2. + etay**2. + etaz**2.)
                   
            v_ra = math.atan2(y_v[i], x_v[i])
            v_dec = np.arcsin(z_v[i]) + r 
            vx = np.cos(v_dec) * np.cos(v_ra)
            vy = np.cos(v_dec) * np.sin(v_ra)
            vz = np.sin(v_dec)
            xix = - x_v[i] + vx 
            xiy = - y_v[i] + vy
            xiz = - z_v[i] + vz
            xi = np.array([xix, xiy, xiz])/np.sqrt(xix**2. + xiy**2. + xiz**2.)

            ax.plot(x_v[i] + r * np.cos(theta) * eta[0] + r * np.sin(theta) * xi[0] , y_v[i] + r * np.cos(theta) * eta[1] + r * np.sin(theta) * xi[1] ,z_v[i] + r * np.cos(theta) * eta[2] + r * np.sin(theta) * xi[2] , color = 'black', linewidth = 0.5  )        

        #ax.scatter(randomns[:, 0], randomns[:, 1], randomns[:, 2], s = 0.2, linewidth = 0.5, color = 'red')
        #ax.scatter(centers[:10000, 0], centers[:10000, 1], centers[:10000, 2], s = 0.5, linewidth = 0.5, color = 'red')
        #ax.scatter( vor_vertices[:, 0], vor_vertices[:, 1], vor_vertices[:, 2],  s = 0.1,linewidth = 0.01, color = 'green' )
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')

        ax.view_init(elev= ori[o][0], azim=ori[o][1])
        plt.savefig("/renoir/renanbos/stage0v4_void_catalogues/stage0mocksv4_voids_DM_dXi_" + str(dXi) + "_delta_t_g_" + str(delta_t_g) + "_zmin_" + str(zmin) + "_zmax_" + str(zmax) + "_orientation_" + str(ori[o][0]) + "_" + str(ori[o][1]) + ".png", dpi = 900)
    zmin = zmax

