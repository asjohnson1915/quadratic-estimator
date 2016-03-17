import correlation
import cosmology
import matplotlib.gridspec as gridspec
import halo
import hod
import os
import kernel
import matplotlib.pyplot as plt
import mass_function
import numpy as np
import matplotlib.pyplot as pyplot
import tools
import Tyche_OQE
from argparse import ArgumentParser
from scipy.interpolate import interp1d

""" This is a module to compute a N(z) distribution using both 
	auto and cross angular correlation functions. This estimate is computed
	using the McQuinn and White optimal quadratic estimator, with corrections
	from Johnson and Blake 16"""

__author__ = ("Andrew Johnson <asjohnson@swin.edu.au>")

parser = ArgumentParser()

parser.add_argument("-f", "--file", dest="myFilenameVariable",
                    help="Mock catalogue file name", metavar="FILE",default='ww_deep_zp0pt50_0pt70.dat')
                    
parser.add_argument("-n", "--num", dest="tomographic_bin",
                    help="tomographic bin num", metavar="INT",default=2)

args = parser.parse_args()

mock_file = args.myFilenameVariable

use_bin = int(args.tomographic_bin)

print('Using bin num',use_bin)

Nz_files = ['nz_z1_kids_binned.dat','nz_z2_kids_binned.dat','nz_z3_kids_binned.dat',
			'nz_z4_kids_binned.dat', 'nz_z4_kids_binned.dat']

Nz_file = os.getcwd() + '/' + 'Nz_data_direct' + '/' + Nz_files[use_bin]
print('Reading Nz file', Nz_file)
Nz_data = np.loadtxt(Nz_file,comments='#')
Nz_direct = {"Pz" : Nz_data[:,1],"redshift" : Nz_data[:,0]}
interp_obj = interp1d(Nz_direct["redshift"],Nz_direct["Pz"],kind='cubic',fill_value=0.00,bounds_error=False)

outputFile_txt = mock_file + '_results.txt'

# Matrix size includes photo-z sample, so is the number of spec-z bins plus one.
# Number of bins constructed to match KiDs-2dflens overlap

degToRad = np.pi/180.0
arcmin_to_rad = 0.000290888
rad_to_arcmins = 1./arcmin_to_rad

default_Nz = {
    "bins_in_z": 30,
    "tot_z_values": 31,
	"matrix_size": 31,
	"redshift_min": 0.005,
	"redshift_max": 1.5,
	"delta_redshift": 0.05
	}

### Both parameters below are set internally in CHOMP 

default_CHOMP = {
	"size_l_array": 200, 
	"size_theta_array": 825
	}
	
### Set relevant parameters for the optimal quadratic estimator  

default_OQE = {
	"max_iterations_schur" : 40,		
	"max_iterations_full" : 8,		
	"abs_accuracy_schur" : 5*10.0**(-3.0), # This is a fraction, total 5% shift, average 0.3% shift
	"abs_accuracy_full" : 2*10.0**(-3.0) 
	}
	
### N_photo_z is the unknown quantity normally but for the simulations it is know 
### and we can use this to compute errors. Nz_spec is know from the spec-z sample

z_bins = default_Nz["bins_in_z"]		
matrix_size = default_Nz["matrix_size"]

Nz_photo = np.empty(z_bins,dtype='float')
Nz_spec = np.empty(z_bins,dtype='float')

b_photo = np.empty(z_bins,dtype='float') 
b_spec = np.empty(z_bins,dtype='float')

### Set bias of both populations (redundent parameters)

b_photo[:] = 1.0  	# Weird scaling needed for CHOMP? Fit is quite good.

b_spec[:] = [0.35, 0.4, 4.49134801e-01, 5.02830481e-01,
   7.41400240e-01,   6.35612655e-01,   6.76130953e-01,   8.26056635e-01,
   7.65606104e-01,   7.40878214e-01,   1.11630740e+00,   1.17123656e+00,
   1.01773324e+00,   1.35083762e+00,   1.61651844e+00,   1.09711158e+00,
   1.38440499e+00,   1.71436925e+00,   1.50014894e+00,   1.27170615e+00,
   1.11981750e+00,   1.75872428e+00,   1.32669148e+00,   1.99501926e+00,
   1.46283866e+00,   1.07034992e+00,   1.15661348e+00,   7.70275310e-01,
   0.7,  0.7]

tot_z = default_Nz["tot_z_values"]	
z_min = default_Nz["redshift_min"]
z_max = default_Nz["redshift_max"]	

### Values at the edge of the bins are

z_values = np.linspace(z_min,z_max,tot_z)

## The central bin values are 

z_mean = (np.roll(z_values,-1)[0:z_bins] + z_values[0:z_bins])/2

sqdeg_to_sqarcmin = 3600.00
sqdeg_to_steradian = 3.0462*10**(-4)

### Properties of spec and photoz samples, compute f_sky.
### Overlap fraction: fraction of galaxies occupying the 
### same halo -- here we assume this to be 1. All 2df galaxies are 
### included in KiDs imaging. 

### The shot noise calcualtion is based on the number of objects 
### per steradian, therefore we need the area of the surveys in 
### units of steradian

### For sky fraction calcualtion f_sky ~ 2.15/41252.96 deg^2 = 0.000052

### Count photo-z will depend on the tomographic bin being investigated. 

Nz_Photo_count = [20838, 17721, 18436,18280, 6376]

print('Using photo-z count', Nz_Photo_count[use_bin])

area_deg = 2.15

sample_properties = {
	"area_phot_deg" : area_deg, 	
	"area_phot" : area_deg*sqdeg_to_steradian, #2.15*sqdeg_to_steradian, 
	"area_spec" : area_deg*sqdeg_to_steradian, #2.15*sqdeg_to_steradian, 
	"overlap_frac" : 1.0,
	"f_sky_sample" : area_deg/41252.96, 
	"count_pz" : Nz_Photo_count[use_bin]
	}

### To Include finite sky coverage one needs to multiply the Fisher matrix by this constant. 
### Quantifies loss of independent modes due to sky coverage.

f_sky = sample_properties["f_sky_sample"]

### bin_count_sz gives the total number of spectroscopic galaxies in a given
### redshifts bins; these values are used to estimate the contribution 
### of shot noise to the Cl power spectra.

bin_count_sz = [
	65.0, 210.0, 528.0, 461.0, 868.0, 547.0, 1206.0,  
	1202.0, 619.0, 577.0, 790.0, 492.0, 610.0, 1550.0,
 	1783.0, 1328.0, 1446.0, 1461.0, 1027.0, 1105.0, 
 	1052.0, 554.0, 593.0, 557.0, 707.0, 433.0, 
 	408.0, 405.0, 187.0, 80.0]

bin_count_sz = 1.0*np.array(bin_count_sz)

### N_photo_z is an unknown quantity (known for mock catalogues only) 
### N_spec_z is the average sky density for the spec-z
### population in each redshift bin

Nz_spec[:] = bin_count_sz[:]/sample_properties["area_spec"]

print('Nz spec sample [galaxies per steradian]', Nz_spec)

### Compute galaxy density [units galaxies per steradian]

dt = np.dtype([('spec',np.float64),('cross',np.float64),('photo',np.float64)])
shot_noise = np.zeros(z_bins,dtype=dt)

sigma_sz = bin_count_sz[:]/sample_properties["area_spec"]
sigma_pz = sample_properties["count_pz"]/sample_properties["area_spec"]

### Compute shot noise for auto and cross correlations
### AJ changed this to make the sizes consistent, check with Chris

shot_noise["spec"][:] = sigma_sz[:]
shot_noise["photo"][:] = sigma_pz

print('Shot noise for spec-z is',shot_noise["spec"][:])

frac = sample_properties["overlap_frac"]

### We use the equation from McQuinn and White 2013 to 
### estimate the shot noise for the cross-correlation Cl's
### the shot noise comes from galaxies existing in the same halo

for bin in range(z_bins):
	pz_shot = shot_noise["photo"][bin]
	sz_shot = shot_noise["spec"][bin]
	shot_noise["cross"][bin] = frac*np.minimum(pz_shot,sz_shot)

### Begin to import all correlation measurements

properties_input_data = {
			"data_size" : 30, 
			"num_bins" : 31,
			"min_arcmin" : 0.3,
			"max_arc_min" : 30.00,
			"min_arcmin_bin" : 0.324, 
			"max_arc_min_bin" : 27.87
			}

print(properties_input_data)

import_dir = os.getcwd() + '/actual_data/' + mock_file

theta_points =  properties_input_data["data_size"]

w_obs = np.zeros([matrix_size,matrix_size,theta_points],dtype=float)
error_corr = np.zeros([matrix_size,matrix_size,theta_points],dtype=float)

### Bins are log-spaced from min to max, N bins so N-1 data points
### Note for calculations want units of radians

### Compute bin edges

min_rad = properties_input_data["min_arcmin"]*arcmin_to_rad
max_rad = properties_input_data["max_arc_min"]*arcmin_to_rad

log_min =  np.log10(min_rad)
log_max =  np.log10(max_rad)

theta_obs_bins = np.logspace(log_min,log_max,properties_input_data["num_bins"])

### Compute bin centres

min_rad = properties_input_data["min_arcmin_bin"]*arcmin_to_rad
max_rad = properties_input_data["max_arc_min_bin"]*arcmin_to_rad

log_min =  np.log10(min_rad)
log_max =  np.log10(max_rad)

theta_obs_mu = np.logspace(log_min,log_max,properties_input_data["data_size"])

print('Data observed at [arcmins]',theta_obs_mu*rad_to_arcmins)
print('Theta Bins are at [arcmins]',theta_obs_bins*rad_to_arcmins)

input_data_old = np.loadtxt(import_dir,comments='#')

input_data = input_data_old[:,1].reshape(1+z_bins*2,theta_points)
error_measurements = input_data_old[:,2].reshape(1+z_bins*2,theta_points)

w_obs[0,0,:] = input_data[0,:]
error_corr[0,0,:] = error_measurements[0,:]

for bin in range(z_bins):
	w_obs[0,bin+1,:] = input_data[1+bin,:]
	w_obs[bin+1,0,:] = input_data[1+bin,:]
	w_obs[bin+1,bin+1,:] = input_data[1+ z_bins+bin,:]

	error_corr[0,bin+1,:] = error_measurements[1+bin,:]
	error_corr[bin+1,0,:] = error_measurements[1+bin,:]
	error_corr[bin+1,bin+1,:] = error_measurements[1+z_bins+bin,:]

### Generate new numpy obs arrays for w_obs,error_corr,theta_obs_mu,theta_obs_bins with the first 5
### elements deleted, this will restrict the angular scales to above 2 acrmins.

w_obs_reduced = np.zeros([matrix_size,matrix_size,theta_points-9],dtype=float)
error_corr_reduced = np.zeros([matrix_size,matrix_size,theta_points-9],dt)

index = [0, 1, 2, 3, 4, 5, 6, 7, 8]

theta_obs_mu_reduced = np.delete(theta_obs_mu,index)
theta_obs_bins_reduced = np.delete(theta_obs_bins,index)

print('Reduced: Data observed at [arcmins]',theta_obs_mu_reduced*rad_to_arcmins)
print('Reduced: Theta Bins are at [arcmins]',theta_obs_bins_reduced*rad_to_arcmins)

for bin in range(z_bins):
	w_obs_reduced[0,bin+1,:] = np.delete(w_obs[0,bin+1,:],index)
	w_obs_reduced[bin+1,0,:] = np.delete(w_obs[bin+1,0,:],index)
	w_obs_reduced[bin+1,bin+1,:] = np.delete(w_obs[bin+1,bin+1,:],index)

	error_corr_reduced[0,bin+1,:] = np.delete(error_corr[0,bin+1,:],index)
	error_corr_reduced[bin+1,0,:] = np.delete(error_corr[bin+1,0,:],index)
	error_corr_reduced[bin+1,bin+1,:] = np.delete(error_corr[bin+1,bin+1,:],index)

### Set cosmology and halo parameters: Parameters set to match Planck cosmology

cosmo_dict = {
    "omega_m0": 0.316,
    "omega_b0": 0.049,
    "omega_l0": 1.0-0.315,
    "omega_r0": 4.15e-5/0.7**2,
    "cmb_temp": 2.726,
    "h"       : 0.6731,
    "sigma_8" : 0.830,
    "n_scalar": 0.965,
    "w0"      : -1.0,
    "wa"      : 0.0
    } 

halo_dict = {
    "stq": 0.3,
    "st_little_a": 0.707,
    "c0": 9.,
    "beta": -0.13,
    "alpha": -1,
    "delta_v": -1
    }

### Initialize the hod object defining how galaxies populate halos. Values used
### in this HOD are from Zehavi et al. 2011 with parameter assumptions from 
### Wake et al. 2011.

hod_dict = {"log_M_min":12.14,
            "sigma":     0.15,
            "log_M_0":  12.14,
            "log_M_1p": 13.43,
            "alpha":      1.0}
            
sdss_hod = hod.HODZheng(hod_dict)

### Initialize the two cosmology objects with at the desired redshifts and
### with the desired cosmological values.

cosmo_single = cosmology.SingleEpoch(redshift=0.0, cosmo_dict=cosmo_dict,with_bao=True)
								 
cosmo_multi = cosmology.MultiEpoch(z_min=0.0,z_max=2.0,cosmo_dict=cosmo_dict,with_bao=True)

### Load Sheth & Tormen mass function 

mass = mass_function.MassFunction(redshift=0.0, cosmo_single_epoch=cosmo_single,
								  halo_dict=halo_dict)

### Initialize the halo object with the mass function and single epoch 
### cosmology implementation is from Seljak2000.

halo_model = halo.Halo(redshift=0.0, input_hod=sdss_hod,
					   cosmo_single_epoch=cosmo_single,
					   mass_func=mass,
					   extrapolate=True)

# Setup arrays to pass to CHOMP to compute both the angular 
# correlation function and the Cl's (can add magnification corrections here)

num_l = default_CHOMP["size_l_array"] 
Cl_ii = np.zeros([z_bins,num_l],dtype='float')  	
l_array = np.zeros(num_l,dtype='float')

num_theta = default_CHOMP["size_theta_array"]
w_ii = np.zeros([z_bins,num_theta],dtype='float')  	
theta_array = np.zeros(num_theta,dtype='float')

### Compute auto-correlations for all spec-z bins (16 bins atm)
### Set precomputed to True after the code has been run once

precomputed = True  

cl_file = 'Cl_array_DEEP.txt'
w_file =  'w_array_DEEP.txt'
theta_file = 'theta_array_DEEP.txt'
ell_file = 'ell_array_DEEP.txt'

if(not precomputed):

	for idx in range(z_bins):
					
		z_mu = (z_values[idx]+ z_values[idx+1])/2.0
		
		print('Using mean',z_mu)		
		print('Using spec-z bin',z_values[idx],z_values[idx+1])
		
		#z_min,z_max,sd = 0.1,0.9,0.05
		#lens_dist = kernel.dNdzGaussian(z_min,z_max,z_mu,sd) 

		#lens_dist = kernel.dNdzGaussian(z_min=0.1,z_max=0.9,z0=0.525,sigma_z=0.05) 
		lens_dist = kernel.dNdz(z_min=z_values[idx],z_max=z_values[idx+1]) 
		#lens_dist = kernel.dNdz(z_min=0.5,z_max=0.55) 
		
		lens_window = kernel.WindowFunctionGalaxy(lens_dist, cosmo_multi) 
		
		### Setup Kernal for integration.
		
		ktheta_min = 10**-20*10**-6*degToRad 
		ktheta_max = 10**5.*180.0*degToRad
		
		con_kernel = kernel.Kernel(ktheta_min=ktheta_min,
							       ktheta_max=ktheta_max,
							       window_function_a=lens_window,
							       window_function_b=lens_window,
							       cosmo_multi_epoch=cosmo_multi,
							       force_quad = True)
	
		### Compute w_theta
		
		corr = correlation.Correlation(
								   theta_min_deg=10**-6,
								   theta_max_deg=180.0,
								   bins_per_decade = 100,
								   input_kernel=con_kernel,
								   input_halo=halo_model,
								   power_spec = 'power_mm',
								   force_quad = True)
		corr.compute_correlation()
				
		w_ii[idx,:] = corr.wtheta_array[:] 
		theta_array = corr.theta_array
							
		### Compute the Cl's
	
		corr_l = correlation.CorrelationFourier(l_min = 1,
											l_max=10**4., 
											input_kernel=con_kernel, 
											input_halo=halo_model,
											power_spec = 'power_mm',
											force_quad = True) 
		corr_l.compute_correlation()

		Cl_ii[idx,:] = corr_l.power_array[:]
		l_array[:] = corr_l.l_array[:]
						
if(not precomputed):

	np.savetxt(cl_file,Cl_ii)
	np.savetxt(ell_file,l_array)
	np.savetxt(w_file,w_ii)
	np.savetxt(theta_file,theta_array)
	
	exit('finished pre-computing correlation functions')
	
if(precomputed):
		
	Cl_ii = np.loadtxt(cl_file,dtype='float')
	l_array = np.loadtxt(ell_file,dtype='float')
	w_ii = np.loadtxt(w_file,dtype='float')
	theta_array = np.loadtxt(theta_file,dtype='float')
	
	
### Calculate the magnification corrections
### These are for Cij, elements, but only for elements with
### source > lens and not for diagonal elements (this contribution is very small)
### Therefore, we need to compute (N^2 - N)/2 elements for Nz = 12, so tot = 66
### Given the defintion of the lens window function here (factor 3/2) 
### the results from Chomp need to be multiplied by a factor of 2. 

### Furthermore, we need to multiple the correlation functions
### by a factor of (1 - alpha). Where alpha is the slope of the number count. 
### Need to be careful with bias parameters here
### as magnification is only sensitive to the bias 
### of the lens (foreground galaxies) 	

### Here we are computing the contributions to just the 
### specz-specz components of the matrix. Therefore 
### alpha is determined solely by the spec-z galaxies. 
		
	include_magnifications = False
	
	if(include_magnifications):
		
		num_l = default_CHOMP["size_l_array"] 
		Cl_ii_mag_corrections = np.zeros([z_bins,z_bins,num_l],dtype='float')  	
		l_array_check = np.zeros(num_l,dtype='float')

		num_theta = default_CHOMP["size_theta_array"]
		w_ii_mag_corrections = np.zeros([z_bins,z_bins,num_theta],dtype='float')  	
		theta_array_check = np.zeros(num_theta,dtype='float')
	
		for bin1 in range(z_bins):
			for bin2 in range(z_bins):
			
				if(bin1 >= bin2): continue
			
				lens_dist = kernel.dNdz(z_values[bin1],z_values[bin1+1])
				source_dist = kernel.dNdz(z_values[bin2],z_values[bin2+1])
		
				print('Lens spec-z bin',z_values[bin1],z_values[bin1+1])
				print('source spec-z bin',z_values[bin2],z_values[bin2+1])
		
				lens_window = kernel.WindowFunctionGalaxy(lens_dist, cosmo_multi) 
				source_window = kernel.WindowFunctionConvergence(source_dist, cosmo_multi)

				### Setup Kernal for integration.
		
				ktheta_min = 10**-20*10**-6*degToRad 
				ktheta_max = 10**5.*180.0*degToRad
		
				con_kernel = kernel.Kernel(ktheta_min=ktheta_min,
									   ktheta_max=ktheta_max,
									   window_function_a=lens_window,
									   window_function_b=source_window,
									   cosmo_multi_epoch=cosmo_multi,
									   force_quad = True)
			
				### Compute the correlation function
	
				corr = correlation.Correlation(theta_min_deg=10**-6,
										   theta_max_deg=180.0,
										   bins_per_decade = 100,
										   input_kernel=con_kernel,
										   input_halo=halo_model,
										   power_spec = 'power_gm',
										   force_quad = True)
				corr.compute_correlation()
	
				w_ii_mag_corrections[bin1,bin2,:] = corr.wtheta_array[:] 
				theta_array_check[:] = corr.theta_array[:]
						
				corr_l = correlation.CorrelationFourier(l_min = 1,
													l_max=10**4., 
													input_kernel=con_kernel, 
													input_halo=halo_model,
													power_spec = 'power_gm',
													force_quad = True) 
				corr_l.compute_correlation()
	
				Cl_ii_mag_corrections[bin1,bin2,:] = corr_l.power_array[:]
				l_array_check[:] = corr_l.l_array[:]
		
### Setup parameters for OQE

max_iter_schur = default_OQE["max_iterations_schur"]
max_iter_full = default_OQE["max_iterations_full"]
		
abs_accuracy_schur = default_OQE["abs_accuracy_schur"]		
abs_accuracy_full = default_OQE["abs_accuracy_schur"]		

Nz_guess = np.zeros(z_bins,dtype=float)
Nz_initial = np.zeros(z_bins,dtype=float)
Nz_true = np.zeros(z_bins,dtype=float)
Pz_initial = np.zeros(z_bins,dtype=float)
dNz = np.zeros(z_bins,dtype=float) 
errors_fisher = np.zeros(z_bins,dtype=float)
errors_fraction = np.zeros(z_bins,dtype=float)

delta_n_by_sigma = np.zeros(z_bins,dtype=float)
delta_n_by_sigma_full = np.zeros(z_bins,dtype=float)

## Estimate of True ##Nz ff

Pz_photo_true = np.zeros(z_bins,dtype=float)
Pz_photo_true = interp_obj(z_mean)
norm = np.sum(Pz_photo_true)
Pz_photo_true = Pz_photo_true/norm

print('Check Pz_guess',Pz_photo_true)
print('check normalized Pz_true',np.sum(Pz_photo_true[:]))

total_density_pz = sample_properties["count_pz"]/sample_properties["area_phot"]

Nz_initial[:] = Nz_guess[:] = sample_properties["count_pz"]/(10.*sample_properties["area_phot"])

Nz_guess[:] = [0.04000, 
 0.04000, 
-0.01231,
 0.00825, 
 0.02805, 
-0.03477, 
 0.06717, 
-0.00097, 
 0.07482, 
 0.01066, 
 0.12264, 
 0.11034, 
 0.14530, 
 0.15308, 
 0.16004, 
 0.01611, 
 0.06502, 
 0.01130, 
 0.15379, 
 0.12154, 
 0.06204, 
-0.08092, 
 0.12290, 
 -0.13416, 
 0.01, 
 0.02302, 
-0.03680, 
 0.04, 
 0.04, 
 0.04000] 

Nz_guess[:] = total_density_pz*Nz_guess[:]

#Pz_initial[:] = Nz_initial[:]/np.sum(Nz_initial)dddffff
#print('check normalized',np.sum(Pz_initial[:]))

### Setup quadratic estimator class and input guess Nz ddd

size_l_array = default_CHOMP["size_l_array"]
size_theta_array = default_CHOMP["size_theta_array"]

### Various approximations can be made here in different circumstances.
### Run Schur approximation until convergence, then run full relation: this 
### includes correlations between bins. Setup new object with updated positions 
### after initial setup has convergence, print out temp results.

outfile_schur = os.getcwd() + '/' + 'new_mocks' + '/' + mock_file.replace('.dat','_results_schur_cross_run2.txt')
outfile_full = os.getcwd() + '/' + 'new_mocks' + '/' + mock_file.replace('.dat','_results_full_cross_run2.txt')

#Optimal_estimator = Tyche_OQE.OQE_Nz(w_obs_reduced,theta_obs_mu_reduced,theta_obs_bins_reduced,w_ii,theta_array,
#						Nz_spec,b_spec,b_photo,Cl_ii,l_array,matrix_size,z_bins,f_sky,
#	   					shot_noise,size_l_array,size_theta_array,use_schur=True,
#  						use_auto_correlations = False,total_density_pz = total_density_pz,error_corr=error_corr)

Optimal_estimator = Tyche_OQE.OQE_Nz(w_obs,theta_obs_mu,theta_obs_bins,w_ii,theta_array,
					Nz_spec,b_spec,b_photo,Cl_ii,l_array,matrix_size,z_bins,f_sky,
 					shot_noise,size_l_array,size_theta_array,use_schur=False,
  					use_auto_correlations = False,total_density_pz = total_density_pz,
  					error_corr=error_corr)

Optimal_estimator.update_dis(Nz_guess[:])

check_errors = True

if(check_errors):

	Optimal_estimator._compute_Amatrix()
	#Optimal_estimator._find_fisher_schur_limt()
	Optimal_estimator._find_schur_parameters()
	Optimal_estimator._find_fisher_matrix()
	errors_fraction[:] = Optimal_estimator.errors_fisher[:]/total_density_pz  ### Hopefully this is right?
	
	print('Updated probability distribution',Optimal_estimator.Pz_photo[:])
	print('Updated Errors on P(z)',errors_fraction[:])
	print('Fractional errors',Optimal_estimator.errors_fisher[:]/Nz_guess[:])
	
	exit()

print('Input N(z) (should tend to a Gaussian)',Nz_guess[:])

## Compute true probability distribution and plot it, should look like a Gaussian.

print('Initial Guess for Probability distribution', Optimal_estimator.Pz_photo[:])

plt_initial_guess = False

if(plt_initial_guess):
	plt.figure(figsize = (8,8)) # set the figure size to be square
	gs = gridspec.GridSpec(1, 1)
	ax = plt.subplot(gs[0, 0])
	plt.plot(z_mean[:],Optimal_estimator.Pz_photo[:],color = 'red',label='$\hat{P}(z_{i})$')
	plt.plot(z_mean[:],Pz_photo_true,color = 'black',label='Fidical $P(z)$')
	ax.set_xlabel(r'$z$ Redshift', fontsize=18,style='italic')
	ax.set_ylabel(r'${P}(z)$', fontsize=18)
	plt.ylim(-0.5,0.5)
	plt.show()

for iter in range(max_iter_schur):

	dNz[:] = Optimal_estimator.find_nz_estimate()
		
	print('check zero bias values have been removed',dNz)
	
	dNz[0] = dNz[1] = dNz[-1] = dNz[-2] = 0.00
	
	Nz_guess[:] = Nz_guess[:] + dNz[:]
		
	print('Computed dN(z) is:',dNz[:])
	print('Computed shift to Pz',dNz[:]/total_density_pz)
	print('New estimate:',Nz_guess[:])
	
	### May need to place bounds on NZ or PZ, to avoid divergences, 
	### it effects other results via A00 calculation, which
	### then chances the Fisher Matrix.
	
	Optimal_estimator.update_dis(Nz_guess[:])
	
	## Code below will update the errors, with the new Fisher matrix from the Nz
	## new error bars are computed
	
	Optimal_estimator._compute_Amatrix()
	Optimal_estimator._find_fisher_schur_limt()	
	
	errors_fraction[:] = Optimal_estimator.errors_fisher[:]/total_density_pz  ### Hopefully this is right?
	
	print('Updated probability distribution',Optimal_estimator.Pz_photo[:])
	print('Updated Errors on P(z)',errors_fraction[:])	
	print('Fractional errors',Optimal_estimator.errors_fisher[:]/Nz_guess[:])
	
	plot_guess = False
	
	if(plot_guess):
		
		Pz_guess_int = interp1d(z_mean,Optimal_estimator.Pz_photo[:], kind='cubic')
		plt.figure(figsize = (8,8)) # set the figure size to be square-ffff gff
		gs = gridspec.GridSpec(1, 1)
		ax = plt.subplot(gs[0, 0])
		plt.plot(z_mean[:],Pz_photo_true,color = 'black',label='Fidical $P(z)$',ls = '--')
		plt.errorbar(z_mean[:], Optimal_estimator.Pz_photo[:],yerr=errors_fraction[:],color='blue',fmt='ro',label='$\hat{P}(z_{i})$')
		#plt.fill_between(z_mean[:], Optimal_estimator.Pz_photo[:]-errors_fraction[:],Optimal_estimator.Pz_photo[:]-errors_fraction[:],facecolor='#b3b3ff',alpha=0.5)
		ax.set_xlabel(r'$z$ Redshift', fontsize=18,style='italic')
		ax.set_ylabel(r'$P(z)$ Schur', fontsize=18)
		ax.axhline(color='k', ls='--') 
		plt.ylim(-0.1,0.4)
		plt.xlim(0.1,1.4)
		
		plt.savefig('Nz-deep-sky-10_deg.pdf')
		plt.close()
		#plt.show()
		
		
	if(np.sum(abs(dNz[:]/total_density_pz)) < abs_accuracy_schur):
		print('Accuracy achieved, leaving loop') 
		break
				
Optimal_estimator.write(outfile_schur,Pz_photo_true,z_mean,errors_fraction,default_Nz["delta_redshift"])

### Start more complication calculation, use end point of previous calculation.

print('Finished Schur approximation, starting full calculation -- convergence criteria increased')

Optimal_estimator_full = Tyche_OQE.OQE_Nz(w_obs,theta_obs_mu,theta_obs_bins,w_ii,theta_array,
					Nz_spec,b_spec,b_photo,Cl_ii,l_array,matrix_size,z_bins,f_sky,
  					shot_noise,size_l_array,size_theta_array,use_schur=False,
   					use_auto_correlations = False,total_density_pz = total_density_pz,error_corr=error_corr)

#Optimal_estimator_full = Tyche_OQE.OQE_Nz(w_obs_reduced,theta_obs_mu_reduced,theta_obs_bins_reduced,w_ii,theta_array,
#						Nz_spec,b_spec,b_photo,Cl_ii,l_array,matrix_size,z_bins,f_sky,
#	   					shot_noise,size_l_array,size_theta_array,use_schur=False,
#  						use_auto_correlations = True,total_density_pz = total_density_pz,error_corr=error_corr)

Optimal_estimator_full.update_dis(Nz_guess[:])

print('Starting guess, full approximation:', Nz_guess[:])

for iter in range(max_iter_full):

	dNz[:] = Optimal_estimator_full.find_nz_estimate()
	
	dNz[0] = dNz[1] = dNz[-1] = 0.00
	
	Nz_guess[:] = Nz_guess[:] + dNz[:]
	
	print('Computed dN(z) is:',dNz[:])
	print('Computed shift to Pz',dNz[:]/total_density_pz)
	print('New estimate is (should tend to a Gaussian):',Nz_guess[:])
	
	Optimal_estimator.update_dis(Nz_guess[:])
	
	## Code below will update the errors, with the new Fisher matrix from the Nz
	## new error bars are computed

	Optimal_estimator_full._find_fisher_schur_limt()
	
	### Is full fisher unrealiable with large Negative values?
	
	errors_fraction[:] = Optimal_estimator_full.errors_fisher[:]/total_density_pz  ### Hopefully this is right?
	
	print('Corrected estimate (fixing negative values):',Nz_guess[:])
	print('Updated probability distribution',Optimal_estimator_full.Pz_photo[:])
	print('Updated Errors',errors_fraction[:])
	
	plot_guess = False
	
	if(plot_guess):
		
		Pz_guess_int = interp1d(z_mean,Optimal_estimator_full.Pz_photo[:], kind='cubic')
		plt.figure(figsize = (8,8)) # set the figure size to be square
		gs = gridspec.GridSpec(1, 1)
		ax = plt.subplot(gs[0, 0])
		plt.plot(z_accurate[:],Pz_true_int(z_accurate),color = 'black',label='Input $P(z)$',ls = '--')
		plt.errorbar(z_mean[:], Optimal_estimator_full.Pz_photo[:],yerr=errors_fraction[:],color='blue',fmt='ro',label='$\hat{P}(z_{i})$')
		ax.set_xlabel(r'$z$ Redshift', fontsize=18,style='italic')
		ax.set_ylabel(r'$P(z)$ FULL', fontsize=18)
		plt.savefig('Nz-deep-sky-10_deg.pdf')
		plt.show()
		
	if(np.sum(abs(dNz[:]/total_density_pz)) < abs_accuracy_full):
		print('Accuracy achieved, leaving loop') 
		break

Optimal_estimator_full.write(outfile_full,Pz_photo_true,z_mean,errors_fraction,default_Nz["delta_redshift"])

exit('Finished Calculation')

### OLD CODE 

### Construct temp, w_obs,theta_obs_mu,theta_obs_bins
### for testing, scale by factor of 100 then 
### see if they are recovered when passed into estimator.
### Only need to generate cross-correlation data.

# N_input_photo = 30.0*Nz_spec[:] 
# w_obs_test_temp = np.zeros([matrix_size,matrix_size,theta_array.size],dtype=float)
# 
# for bin_index in range(z_bins): 
# 
# 	w_obs_test_temp[0,1+bin_index,:] = w_obs_test_temp[1+bin_index,0,:] =  (
# 			(Nz_spec[bin_index]*b_spec[bin_index]*
# 			N_input_photo[bin_index]*b_photo[bin_index])*w_ii[bin_index,:])
# 
# w_ii_obs_test = []
# 
# for zbin in range(z_bins):
# 	interp_obj = interp1d(theta_array,w_obs_test_temp[0,1+zbin,:],kind='linear',fill_value=0.00,bounds_error=False)	
# 	w_ii_obs_test.append(interp_obj)
# 
# w_obs_cross = lambda theta, bin: w_ii_obs_test[bin](theta)
# 
# NEW_SIZE = 5000
# theta_min_deg = 10**-5.
# theta_max_deg = 180.0
# 
# deg_to_rad = np.pi/180.0
# log_min_new = np.log10(theta_min_deg*deg_to_rad)
# log_max_new = np.log10(theta_max_deg*deg_to_rad)
# 
# theta_obs_bins_test = np.logspace(log_min_new,log_max_new,NEW_SIZE+1)
# theta_obs_mu_test = np.logspace(log_min_new,log_max_new,NEW_SIZE)
# 
# w_obs_test = np.zeros([matrix