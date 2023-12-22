import numpy as np
import matplotlib.pylab as plt
import scipy.integrate
import time as time
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import pandas as pd
from matplotlib import rc
from scipy import interpolate
import sys
import numba
from numba import jit
from astropy.io import fits
from astropy.table import Table
from sklearn.neighbors import KDTree
from PyAstronomy import pyasl
import math
import os
from os.path import exists
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


################## INPUTS #######################
# Path in which the catalogs can be found:
#path_voids = '/Users/renanbos/Desktop/Void_Lensing_project/voids/'
#path_sources = '/datadec/desi/c3wg/stage1mocks/'
#path_twod_voids = '/renoir/renanbos/Void_Lensing_proj/Results_tracer_BGSdelta_t_0.2/'
# Choose the tracer:
tracer = "BGS"
#tracer = "LRG"

path_sources = '/datadec/desi/c3wg/stage0mocks_catalogues/'

rand = 0
new_catalogs = 0
Pauline_catalogs = 1

path_twod_voids = '/renoir/renanbos/stage0v4_void_catalogues/'
#path_twod_voids = '/renoir/renanbos/New_catalogs_mac_/'
# Whether using random voids or not:
#rand = 0

# Delta_t:
delta_t = 0.5

dXi = 100

# Footprint:
ra_min = 0
ra_max = 180
dec_min = 0
dec_max = 90

# Maximum and minimum redshift in lenses catalog:
z_min = 0.1
z_max = 0.3
z_mean =  (z_max + z_min)/2.

# Minimum and maximum redshift for sources:
zs_min = 0.5
zs_max = 1.5

# Photo survey:
survey = 'des'
#survey = 'hsc'
#survey = 'kids'

# Sources redshift:
#redshift_range_sources = np.array(["zp0pt63_0pt90"])
#redshift_range_sources = np.array(["zp0pt20_0pt43", "zp0pt43_0pt63", "zp0pt63_0pt90", "zp0pt90_1pt30" ])
redshift_range_sources = np.array([ "zp0pt5_0pt7", "zp0pt7_0pt9", "zp0pt9_1pt1", "zp1pt1_1pt5"])
#redshift_range_sources = np.array([ "zp0pt5_0pt7"])


# Galaxy Shape noise:
sigma_shape = 1.

# Minimum and maximum void radius in the stack:
rmin_v = 0.
rmax_v = 100.
# Which catalog to use in the case of WL signal estimation:
#lenses_catalog = "Summary_lenses_"+tracer+"_"+redshift_range_lens+".txt"

# Wether to use 2D voids or not:
twod_voids = 1

# Binning:
#binning = 'log'
binning = 'linear'

# Whether apply lower (select void with radius smaller that radius_cut) cut or not:
bin_cut = False
radius_min = 10 # in Mpc/h
radius_max = 100
lower_cut = 1
higher_cut = 100

# Cosmology:
Om = 0.286
Ol = 1 - Om
H0 = 70.
h = H0/100.

# Physical constants:
c = 9.72e-09 # This is in pc/s
G = 4.49e-30 # This is in pc^3 M0^-1 s^-2

# Output name:
#out_file = "corr_void_gal_"+tracer+"_"+redshift_range_lens+".dat"

#data=np.loadtxt(path_sources+sources_catalog, skiprows=3)
t_s = time.time()
reg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

sources_list = []

print ("Importing sources ...")

for i in range(len(reg)):
    for j in range(len(redshift_range_sources)):
        data = Table.read(path_sources + 'stage0mockv4_mock0_reg' + str(reg[i]) + '_sources_' + redshift_range_sources[j] + '.fits', format='fits')
        catalog = data.to_pandas()
        sources_list.append(catalog)

data_sources = np.vstack( sources_list )

sources =  pd.DataFrame(data_sources, columns = ['ra','dec', 'z_spec', 'z_phot', 'gamma_1', 'gamma_2', 'e_1', 'e_2', 'wei'])


w = (sources['ra'] > ra_min) & (sources['ra'] < ra_max) & (sources['dec'] > dec_min) & (sources['dec'] < dec_max) 
sources = sources[w] 

#print ("min and max sources dec: ", np.min(np.radians(sources['dec'])), np.max(np.radians(sources['dec'])))
print ("min and max sources ra: ", np.min(np.radians(sources['ra'])), np.max(np.radians(sources['ra'])))

sources['x'] = np.cos(np.radians(sources['dec'])) * np.cos(np.radians(sources['ra']))
sources['y'] = np.cos(np.radians(sources['dec'])) * np.sin(np.radians(sources['ra']))
sources['z'] = np.sin(np.radians(sources['dec']))


sources['sin_dec'] = np.sin(np.radians(sources['dec']))
sources['cos_dec'] = np.cos(np.radians(sources['dec']))

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

cat = []
zmin = z_min
while z_interp( r_c_interp(zmin) + dXi ) <= z_max:
 
    zmax = z_interp( r_c_interp(zmin) + dXi )
    
    zmean = zmin + (zmax - zmin)/2.
    print ("Appending catalog with average redshift ", zmean)
    if not exists('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + str(np.round(zmin, 3)) + "_zmax_" + str(np.round(zmax, 3)) + "_last.dat"):
        print ("It does not exist for zmin and max ", zmin, zmax)
        st1 =  str(np.round(zmin, 3) - 0.001)
        print ("It does not exist for zmin and max ", zmin, zmax)
        if not exists('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + st1 + "_zmax_" + str(np.round(zmax, 3)) + "_last.dat"):
            st2 =  str(np.round(zmax, 3) - 0.001)
            if not exists('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + str(np.round(zmin, 3)) + "_zmax_" + st2 + "_last.dat"):
                cat_ = np.loadtxt('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + st1 + "_zmax_" + st2 + "_last.dat")
    else:
        cat_ = np.loadtxt('/renoir/renanbos/stage0v4_void_catalogues/' + "2D_catalog_stage0v4_dXi_" + str(dXi) + "_delta_t_" + str(delta_t) + "_zmin_" + str(np.round(zmin, 3)) + "_zmax_" + str(np.round(zmax, 3)) + "_last.dat")

    cat.append( np.column_stack( ( cat_ , zmean * np.ones(len(cat_))) ))
    zmin = zmax

lenses = np.concatenate(cat)
lenses = pd.DataFrame(lenses, columns = ['x', 'y', 'z', 'radius', 'redshift'])

lenses_ra = np.zeros(len(lenses))
lenses_dec = np.zeros(len(lenses))
for i in range(len(lenses_ra)):
    lenses_ra[i] = math.atan2(lenses['y'][i], lenses['x'][i])
    lenses_dec[i] = np.arcsin(lenses['z'][i])
lenses['ra'] = lenses_ra
lenses['dec'] = lenses_dec

lenses['ra[deg]'] = 180. * lenses['ra']/np.pi
lenses['dec[deg]'] = 180. * lenses['dec']/np.pi

print ("Min and max ras of lenses: ", np.min(lenses['ra[deg]']), np.max(lenses['ra[deg]']))
print ("Min and max decs of lenses: ", np.min(lenses['dec[deg]']), np.max(lenses['dec[deg]']) ) 

w = (lenses['radius'] * r_c_interp(lenses['redshift']) > radius_min) & (lenses['radius'] * r_c_interp(lenses['redshift']) < radius_max) & (lenses['ra[deg]'] > ra_min) & (lenses['ra[deg]'] < ra_max) & (lenses['dec[deg]'] > dec_min )  & (lenses['dec[deg]'] < dec_max)

lenses = lenses[w]

print ("The numer of voids will be: ", len(lenses))
print ("The first void will be: ", np.array(lenses)[0])
print ("lenses", lenses)

# Calculate the comoving distances:
def H(z):
    return H0*np.sqrt(Om*(1+z)**3.+Ol)

z_vec = np.linspace(0.01, 1.7, 2000)
r_c = np.ones(len(z_vec))
# Angular diameter distance function: ATENTION: this is in Mpc/h 
for i in range(len(z_vec)): 
    r_c[i] = 2997.92 * scipy.integrate.simps(1./np.sqrt( Om*( 1.+np.linspace(0., z_vec[i], 1000) )**3.+Ol ) , np.linspace(0., z_vec[i], 1000)) 

sigma_const = (c**2.)/(4.*np.pi*G)

sources['comov_dist'] = np.interp(sources['z_phot'],z_vec, r_c)
lenses['comov_dist'] = np.interp(lenses['redshift'],z_vec, r_c)

print(" You are going to stack ", len(lenses), " voids found in slices of ", dXi )

# Choose the binning:
rmin = 0.0
rmax = 5.0
nbins = 30

# Indexing:

ra = 0
dec = 1
z_phot = 3
gamma_1 = 4
gamma_2 = 5
weight_index = 8
x_sources = 9
y_sources = 10
z_sources = 11
sin_dec = 12
cos_dec = 13
comov_dist = 14
e_1 = 6
e_2 = 7

# Find sources inside the void:
@jit(nopython=True)
def red_dist_void_sources( ra_l, ra_s, dec_l, dec_s ):
    x_l = np.cos(dec_l) * np.cos(ra_l)
    y_l = np.cos(dec_l) * np.sin(ra_l)
    z_l = np.sin(dec_l)
    x_s = np.cos(dec_s) * np.cos(ra_s)
    y_s = np.cos(dec_s) * np.sin(ra_s)
    z_s = np.sin(dec_s)
#    print ("x_l, y_l, z_l: ", x_l, y_l, z_l)
#    print ("", )

    return np.arccos(np.dot( np.column_stack( (x_s, y_s, z_s) ), np.array([x_l, y_l, z_l]) ))
#    return np.arccos( np.cos( ra_s - ra_l  ) * np.cos( dec_l ) * np.cos( dec_s ) + np.sin(dec_l ) * np.sin( dec_s ) ) 

@jit(nopython=True)
def index_fun(red_proj_dist, rmin, rmax, nbins, lens_z, ra_diff, dec_diff, sources_z, sources):
    index = np.floor( (red_proj_dist - rmin)/(rmax - rmin)*nbins)
    w = (index >= 0) & (index < nbins) & (sources_z > lens_z)    
    return sources[w], ra_diff[w], dec_diff[w], index[w]

@jit(nopython=True)
def sigma_crit_fun(sigma_const, lens_z, comov_dist_sources, comov_dist_lens, comov_dist_diff):
#    return (sigma_const * (1./(1.+lens_z))  * comov_dist_sources/comov_dist_lens/comov_dist_diff )/1e6
    return (sigma_const * comov_dist_sources/comov_dist_lens/comov_dist_diff )/1e6

@jit(nopython=True)
def sin_cos_fun(ra_s , dec_s, ra_l, dec_l, dist):
    cos_phi =  np.sin( ra_s - ra_l ) * np.cos(dec_s)/np.sin(dist)
    sin_phi = (np.cos( dec_s ) * np.sin(dec_l) - np.sin(dec_s) * np.cos( dec_l ) * np.cos( ra_s - ra_l) )/np.sin(dist)
    return sin_phi, cos_phi      
      
#@jit(nopython=True)
def histograms(red_proj_dist, weight, sigma_crit, gamma_t, gamma_x, sources_weight, sources_e_rms, sources_m, nbins, rmin, rmax):
    sum_signal_t = np.histogram(red_proj_dist, weights = weight * sigma_crit * gamma_t ,bins = nbins, range = (rmin, rmax))[0]
    sum_weight_t_x = np.histogram(red_proj_dist, weights = weight, bins = nbins, range = (rmin, rmax))[0]
    sum_signal_x = np.histogram(red_proj_dist, weights = weight * sigma_crit * gamma_x ,bins = nbins, range = (rmin, rmax))[0]
 
    return sum_signal_t, sum_weight_t_x, sum_signal_x  

@jit(nopython=True) # This function must receive values in degrees:
def select_sources( lens_ra, lens_dec, ra_s, dec_s):
    w_s = ( ra_s > lens_ra - 50) & (ra_s < lens_ra + 50) & (dec_s > lens_dec - 50) & (dec_s < lens_dec + 50)
    return w_s

#numb_sources = [] # list that will store the number of sources in each bin of each void

lens_ra = np.array(lenses['ra'])[0]
lens_dec = np.array(lenses['dec'])[0]
lens_redshift = np.array(lenses['redshift'])[0]
#    lens_radius = np.radians(np.array(lenses['radius'])[i])
lens_radius = np.array(lenses['radius'])[0]
lens_comov_dist = np.array(lenses['comov_dist'])[0]

def get_sigma(i): 
   
    ti = time.time()

    sum_signal = np.zeros(nbins)
    sum_weight = np.zeros(nbins)
    ti1 = time.time()
    ti = time.time()

    lens_ra = np.array(lenses['ra'])[i]
    lens_dec = np.array(lenses['dec'])[i]
    lens_redshift = np.array(lenses['redshift'])[i]
#    print ("Estimating DSMD using void at redshift: ", lens_redshift)

    lens_radius = np.array(lenses['radius'])[i]
    lens_comov_dist = np.array(lenses['comov_dist'])[i]
  
    #-- Transverse separation in the plane of the lens
    ti = time.time()
    w_s = select_sources(180 * lens_ra/np.pi, 180 * lens_dec/np.pi,  sources_[:, ra], sources_[:, dec])
    ra_s_ = sources_[:, ra][w_s]
    dec_s_ = sources_[:, dec][w_s]

    tf = time.time()
    if i == 0 or i == 5:
        print ("Time for selecting sourcers: ", tf - ti)

    if i == 0 or i == 5:
        print ("Calculating distances ...")
    ti_dist = time.time()
#    dist = red_dist_void_sources(np.radians(lens_ra), np.radians(ra_s_), np.radians(lens_dec), np.radians(dec_s_) )
    dist = red_dist_void_sources(lens_ra, np.radians(ra_s_), lens_dec, np.radians(dec_s_) )

#    dist = np.radians(pyasl.getAngDist(lens_ra * 180/np.pi, lens_dec * 180/np.pi, ra_s , dec_s )) 
    tf_dist = time.time()
    if i == 3 or i == 0 or i == 5:
        print ("lens ra and dec: ", lens_ra, lens_dec)
        print ("lens radius in comoving distance: ", np.sin(lens_radius) * lens_comov_dist)
        print ("Time for computing red_proj_dist: ", tf_dist - ti_dist, np.sin(np.sort(dist)) * lens_comov_dist)
        print ("radius in reduced distance: ", np.sort(dist)/lens_radius)


    sources_comov_dist = sources_[:,comov_dist][w_s]
    comov_dist_diff = sources_comov_dist - lens_comov_dist
    
    ti_sc = time.time()
    sigma_crit = sigma_crit_fun(sigma_const, lens_redshift, sources_comov_dist,lens_comov_dist, comov_dist_diff)
    tf_sc = time.time()

    ti_sc = time.time()
    result_sin_cos = sin_cos_fun( np.radians(ra_s_) , np.radians(dec_s_), lens_ra, lens_dec, dist)
    tf_sc = time.time()
    if i == 3:
        print ("Time for computing sin and cos: ", tf_sc - ti_sc, result_sin_cos)
    cos_phi = result_sin_cos[1]
    sin_phi = result_sin_cos[0]
    
    cos_2phi = cos_phi**2. - sin_phi**2.
    sin_2phi = 2 * sin_phi * cos_phi

    weight  = ((1./sigma_crit)**2.) 

    g_1 = sources_[:,gamma_1][w_s]
    g_2 = sources_[:,gamma_2][w_s]
    gamma_x = - (g_2 * cos_2phi + g_1 * sin_2phi)
    gamma_t = - (g_1 * cos_2phi - g_2 * sin_2phi)
 
    ti_hist = time.time()
    sum_signal_t = np.histogram(dist/lens_radius, weights = weight * sigma_crit * gamma_t ,bins = nbins, range = (rmin, rmax ))[0]
    sum_weight_t_x = np.histogram(dist/lens_radius, weights = weight, bins = nbins, range = (rmin, rmax ))[0]
    sum_signal_x = np.histogram(dist/lens_radius, weights = weight * sigma_crit * gamma_x ,bins = nbins, range = (rmin, rmax ) )[0]

    tf_hist = time.time()
    if i == 3:
        print ("Time for computing histograms: ", tf_hist - ti_hist)
 #   sum_signal_t = np.histogram(red_proj_dist, weights = weight * sigma_crit * gamma_t ,bins = nbins, range = (rmin, rmax))[0]
 #   sum_weight_t_x = np.histogram(red_proj_dist, weights = weight, bins = nbins, range = (rmin, rmax))[0]
 #   sum_signal_x = np.histogram(red_proj_dist, weights = weight * sigma_crit * gamma_x ,bins = nbins, range = (rmin, rmax))[0]
 

  #  sum_R = np.histogram(red_proj_dist, weights = sources[:, weight_index] * sources[:, e_rms]**2., bins = nbins, range = (rmin, rmax))[0]
  #  sum_weight = np.histogram(red_proj_dist, weights = sources[:, weight_index] , bins = nbins, range = (rmin, rmax))[0]

  #  sum_m = np.histogram(red_proj_dist, weights = sources[:, weight_index] * sources[:, m], bins = nbins, range = (rmin, rmax))[0]

    tf = time.time()
    if i == 0 or i ==5 or i == 10 or i == 20 or i == 50:
        print ('Time for one void: ', tf - ti, ' Estimated time for all voids: ', (len(lenses)*(tf - ti))/3600., 'h')
        print ('sum_signal :', sum_signal_t)
#    return sum_signal_t, sum_weight_t_x, sum_signal_x, sum_R, sum_m, sum_weight  
    return sum_signal_t, sum_weight_t_x, sum_signal_x 

r_rv = (0.5 + np.arange(nbins)) * (rmax - rmin)/nbins + rmin
delta_sigma_num_t = np.zeros(nbins)
delta_sigma_den_t = np.zeros(nbins)
delta_sigma_num_x = np.zeros(nbins)
delta_sigma_den_x = np.zeros(nbins)
#R_bar = np.zeros(nbins)
#m_bar = np.zeros(nbins)
#weight = np.zeros(nbins)


# Split into two sources samples:
zs_max = np.max(sources['z_spec'])
zs_min = np.min(sources['z_spec'])

zs_1 = zs_min + (zs_max - zs_min)/4.
zs_2 = zs_min + 2 * (zs_max - zs_min)/4.
zs_3 = zs_min + 3 * (zs_max - zs_min)/4.
zs_4 = zs_max

w1 = (sources['z_spec'] < zs_2) & (sources['z_spec'] > zs_1)
w2 = (sources['z_spec'] < zs_3) & (sources['z_spec'] > zs_2)
w3 = (sources['z_spec'] < zs_4) & (sources['z_spec'] > zs_3)

sources_1 = sources[w1]
sources_2 = sources[w2]
sources_3 = sources[w3]

sources_all = [sources_1, sources_2, sources_3]
sources_w = [len(sources_1), len(sources_2), len(sources_3)]

results_all = []
for i in range(len(sources_all)):
    print ("Estimating using sources ", i)
    print ("The number of sources is: ", sources_w[i])
    sources_ = np.array(sources_all[i])

    # Use multiprocessing for calling the funciton multiple times:
    context = multiprocessing.get_context('fork')

    #pool = context.Pool(processes= multiprocessing.cpu_count()  )
    pool = context.Pool(processes = 40, maxtasksperchild = 10) 

    t1 = time.time()
    results_all.append(pool.map(get_sigma, np.arange(len(lenses))))
    pool.close()
    t2 = time.time()
#print ("results :", results[:][4])

for i in range(len(lenses)):
    for j in range(len(results_all)):    
            delta_sigma_num_t += sources_w[j] * results_all[j][i][0]/np.sum(sources_w)
            delta_sigma_den_t += sources_w[j] * results_all[j][i][1]/np.sum(sources_w)
            delta_sigma_num_x += sources_w[j] * results_all[j][i][2]/np.sum(sources_w)
            delta_sigma_den_x += sources_w[j] * results_all[j][i][1]/np.sum(sources_w)

#    R_bar += results[i][3]

#R_bar_ = 1. - R_bar/weight
#m_bar_ = m_bar/weight
#delta_sigma_final_t = ( 1./(2 * R_bar_ * (1. + m_bar_) ) ) * delta_sigma_num_t/delta_sigma_den_t
#delta_sigma_final_x = ( 1./(2 * R_bar_ * (1. + m_bar_) ) ) * delta_sigma_num_x/delta_sigma_den_x                     
delta_sigma_final_t = delta_sigma_num_t/delta_sigma_den_t
delta_sigma_final_x = delta_sigma_num_x/delta_sigma_den_x                     

print("delta_sigma_final_t: ", delta_sigma_final_t)
print("delta_sigma_final_x: ", delta_sigma_final_x)

print ("Saving results ...")
#    np.savetxt("/renoir/renanbos/Void_Lensing_proj/Delta_sigma_2D_random_"+np.str(rand)+"_"+tracer+"_nz_"+np.str(n_z)+"_z_min_"+np.str(z_min)+"_z_max_"+np.str(z_max)+"_delta_t_0.2_lower_cut_"+np.str(lower_cut)+"_higher_cut_"+np.str(higher_cut)+"_e_cal_NUMB_SOURCERS.dat", numb_sources_mean)
np.savetxt("/renoir/renanbos/Void_Lensing_proj/Delta_Sigma_stage0v4/Delta_sigma_t_2D_My_new_voids_stage0v4_"+tracer+"_dXi_"+str(dXi)+"_z_min_"+ str(z_min)+"_z_max_"+ str(z_max)+"_delta_t_"+ str(delta_t)+"_radius_min_" + str(radius_min)+ "_radius_max_" + str(radius_max)+ "_" +  str(survey) + "_ra_min_" +  str(ra_min) + "_ra_max_" +  str(ra_max) + "_dec_min_" +  str(dec_min) + "_dec_max_" +  str(dec_max) + "_s.dat", np.transpose(np.vstack((r_rv ,delta_sigma_final_t))))
np.savetxt("/renoir/renanbos/Void_Lensing_proj/Delta_Sigma_stage0v4/Delta_sigma_x_2D_My_new_voids_stage0v4_"+tracer+"_dXi_"+str(dXi) +"_z_min_"+ str(z_min)+"_z_max_"+ str(z_max)+"_delta_t_"+ str(delta_t)+"_radius_min_" + str(radius_min)+ "_radius_max_" + str(radius_max)+ "_" +  str(survey) + "_ra_min_" +  str(ra_min) + "_ra_max_" +  str(ra_max) + "_dec_min_" +  str(dec_min) + "_dec_max_" +  str(dec_max) + "_s.dat", np.transpose(np.vstack((r_rv ,delta_sigma_final_x))))


'''
print ("Plotting ...")  
# Plot the Differential surface mass density:
plt.figure()
#plt.plot(10**r_rv, delta_sigma_final)s
plt.plot(r_rv, delta_sigma_final_t, label = r"$\Sigma_{t}$")
plt.plot(r_rv, delta_sigma_final_x, label = r"$\Sigma_{x}$", color = 'black')
plt.legend()
plt.grid(True)
#plt.xscale("log")
#plt.title(np.str(redshift_range_lens)+" "+np.str(tracer))
plt.title("$" + tracer + "$ "+" "+r"$\bar{z} = $" + "$ " + np.str(np.round(z_mean, 2)) + "$")
plt.xlabel(r"$R_{p}/R_{v}$", fontsize = 15)
plt.ylabel(r"$\Delta \Sigma (R_{p}/R_{v}) [M_{\odot}h/pc^{2}] $", fontsize = 15)#plt.savefig("/renoir/renanbos/Void_Lensing_proj/Delta_Sigma_final_rmin_"+np.str(rmin_v_data)+"_rmax_"+np.str(rmax_v_data)+tracer+"new_HERE_"+np.str(nbins)+"_"+binning+"_optimized_stage1_hsc_weight_SN"+np.str(SN)+"_MN"+np.str(MN)+".pdf")
#plt.show()
if bin_cut:
    plt.savefig("/renoir/renanbos/Void_Lensing_proj/Delta_sigma_2D_"+tracer+"_nz_"+np.str(n_z)+"_z_min_"+np.str(z_min)+"_z_max_"+np.str(z_max)+"_delta_t_"+np.str(delta_t)+"_lower_cut_"+np.str(lower_cut)+"_higher_cut_"+np.str(higher_cut)+"_e_cal.pdf")
else:
    plt.savefig("/renoir/renanbos/Void_Lensing_proj/Delta_sigma_2D_"+tracer+"_nz_"+np.str(n_z)+"_z_min_"+np.str(z_min)+"_z_max_"+np.str(z_max)+"_delta_t_"+np.str(delta_t)+"_radius_cut_higher_"+np.str(radius_cut)+"_e_cal_new_catalog.pdf")
'''
print (" Done!")




