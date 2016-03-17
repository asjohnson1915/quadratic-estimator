import math
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
from misc import Pl_interpolator
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import time

"""OPEN JOBS SECTION:
1. Include magnification bias
2. Remove all plot statements and Run and check converagence
3. Add auto correlations. Run and check converagence
"""

arcmin_to_rad = 0.000290888
rad_to_arcmin = (1.0/arcmin_to_rad)

__author__ = ("Andrew Johnson <asjohnson@swin.edu.au>")

class OQE_Nz(): 

	"""Class for calculating Nz for photo-z sample using angular 
	cross-correlations. This is done with a quadratic estimator
	
	Args:
	w_obs: matrix of observed angular correlation functions
	theta_obs_mu: mean values of theta for measured correlation function (size N-1)
	theta_obs_bins: bin values of theta for measured correlation function (size N)
	w_ii: theory predictions for auto-correlations of w_ii
	theta_array: theta values where w_ii has been computed 
	Nz_spec: The Nz for the input spec-z populations
	b_spec: bias of the spec_z population
	b_photo: bias of the photo_z population
	Cl_ii: theory predictions for the CL's (auto-correlations)
	l_array: l values where Cl's have been computed
	matrix_size: size of the A matrix, bin number + 1 
	z_bins: z-bin values of spec-z sample
	f_sky: sky fraction of sample (?)
	shot_nose: array of shot noise values for spec, photo statistics"""
    
	def __init__(self,w_obs,theta_obs_mu,theta_obs_bins,w_ii,
    			theta_array,Nz_spec,b_spec,b_photo,Cl_ii,l_array,
    			matrix_size,z_bins,f_sky, shot_nose,size_l_array,
    			size_theta_array,use_schur,use_auto_correlations,
    			total_density_pz,error_corr):
    
		self.w_obs = w_obs
		self.theta_obs_mu = theta_obs_mu
		self.theta_obs_bins = theta_obs_bins
		self.w_ii = w_ii
		self.theta_array = theta_array
		self.Nz_spec = Nz_spec
		self.b_spec = b_spec
		self.b_photo = b_photo
		self.Cl_ii = Cl_ii
		self.l_array = l_array
		self.matrix_size = matrix_size
		self.z_bins = z_bins
		self.f_sky = f_sky
		self.shot_nose = shot_nose
		self.size_l_array = size_l_array
		self.size_theta_array = size_theta_array
		self.error_corr = error_corr
		
		self.total_density_pz = total_density_pz
		
		self.use_schur = use_schur
		self.use_auto_correlations = use_auto_correlations

		if(self.use_schur and self.use_auto_correlations):
			exit(':(')
		
		### Probability distributions for w_theta theory calculation. Note
		### does not need actual N_z, only Pz -- different from Cl's
		
		self.Pz_spec = np.zeros(self.z_bins,dtype=float)
		self.Pz_photo = np.zeros(self.z_bins,dtype=float)
		
		### The arrays below are computed using 
		### the functions in this class and are used throughout

		### For summation over P_l(theta) below we need an l array of integers
		### we also need this vector to pass to the Pl_interpolator object.
		### Finally, this also  defines integration range over ell:
		### we don't need to integrate over all ell given the weights.
		### This is no quite consistent. Need to fix range

		self.l_max,self.l_min = 10**4-2.0,10**1.+ 2.0			
		self.l_int_vec = np.arange(int(self.l_min) - 1,int(self.l_max) + 1,dtype='float')
	
		### Note, if you change l_int_vec need to rerun Pl calculation 
		### l_min,l_max range need to be smaller than vec range, given one sets 
		### up interpolation and the other sets integration limits.
	
		self.Nz_photo_guess = np.zeros(self.z_bins,dtype=float)
		self.A_matrix = np.zeros([self.matrix_size,self.matrix_size,self.l_int_vec.size],dtype=float)
		self.cross_cor_temp = np.zeros([self.l_int_vec.size,self.z_bins],dtype=float)  					  
		self.schur_p_temp = np.zeros(self.l_int_vec.size,dtype=float)
		self.fisher_nz_inverse = np.zeros([self.z_bins,self.z_bins],dtype=float)
		self.errors_fisher = np.zeros(self.z_bins,dtype=float)

		self.deg_to_rad = np.pi/180.0
		self.Cl_ii_accurate = np.zeros([self.z_bins,self.l_int_vec.size])
		self.w_ii_predict = np.zeros([self.matrix_size,self.matrix_size,self.theta_array.size],dtype='float')  

		### Arrays to store various weights

		self.vi_l = np.zeros([self.z_bins,self.l_int_vec.size],dtype=float)
		self.g_jk_l = np.zeros([self.z_bins,self.z_bins,self.l_int_vec.size],dtype=float)
		self.fj_l = np.zeros([self.z_bins,self.l_int_vec.size],dtype=float)
		self.kj_l = np.zeros([self.z_bins,self.l_int_vec.size],dtype=float)
		
		### Need to setup interpolation for all Cl measurements, then 
		### change to a finer delta l grid, this is because we 
		### need to sum over l for later calculations
		
		Cl_ii_interp = []
		log_l_vec = np.log10(self.l_array)
		
		for zbin in range(self.z_bins):
			interp_obj = InterpolatedUnivariateSpline(np.log10(self.l_array),np.log10(self.Cl_ii[zbin,:]))
			Cl_ii_interp.append(interp_obj)
		
		self.Cl_ii_fn = lambda l, bin: np.power(10, Cl_ii_interp[bin](np.log10(l)))
					
		for zbin in range(self.z_bins):
			for l_index,l in enumerate(self.l_int_vec):
				self.Cl_ii_accurate[zbin,l_index] = self.Cl_ii_fn(l,zbin)
		
# 		fig = pyplot.plot(self.l_int_vec[:],self.Cl_ii_accurate[0,:],color='yellow')
# 		fig = pyplot.plot(self.l_array,self.Cl_ii[0,:],color='orange')
# 		pyplot.ylabel('C_ii accurate')
# 		pyplot.xlabel('l')
# 		pyplot.yscale('log')
# 		pyplot.xscale('log')
# 		pyplot.show()
# 		pyplot.savefig('plots/Cl_z0.pdf')
						
		### Relevant parameters for integration over theta
		
		self.theta_properties = {
				"num_theta_bins" : 10**4,
				"theta_min_deg" : 10**-5.,
				"theta_max_deg" : 180.0,
				"kernel_bessel_limit_local" : 8,
				"theta_approx_cut" : 1*10**(-3) 
				}
			
		### Need a more accurate theta array to setup interpolation for Pl's
		### and to use as grid for conversion to config space (see below)
		
		bins = 10**4
		rad_min = self.theta_properties["theta_min_deg"]*self.deg_to_rad
		rad_max = self.theta_properties["theta_max_deg"]*self.deg_to_rad
		log_theta_min = np.log10(rad_min)
		log_theta_max = np.log10(rad_max)
		
		self.theta_array_accurate = np.logspace(log_theta_min,log_theta_max,self.theta_properties["num_theta_bins"])
		
		### Setup interpolator for P_l(cos(theta)) this should only be computed once. 
		### Once computed set Import = True to import a grid from a saved .dat file
		### Re-run if the grid passed to the interpolated is a different range
		### Importing saved file may introduce error
		
		self.Pls_import = Pl_interpolator(self.theta_array_accurate,
								self.l_int_vec,Import = True,Save = False)
		
		print('Setting up interpolation')
								
		self.Pls_import.interp_setup_2D()	
		
	def update_dis(self,Nz_update):
		self.Nz_photo_guess = Nz_update
		
		### Probability distributions for w_theta theory calculation. Note
		### does not need actual N_z, only Pz -- different from Cl's
		
		self.Pz_spec[:] = 1.0 				# Localised in a single redshift bin
		
		#self.Pz_photo[:] = self.Nz_photo_guess[:]/(np.sum(self.Nz_photo_guess[:])) # Ratio of z's in bin to all z's
		
		### Dont impose normalisation		
		self.Pz_photo[:] = self.Nz_photo_guess[:]/(self.total_density_pz) # Ratio of z's in bin to all z's
	
		print('Check normalized',np.sum(self.Pz_photo[:]))	
			
	def find_nz_estimate(self):
		""" Main function"""
		 
		temp_vec = np.zeros(self.z_bins,dtype=float)
		correction_vec = np.zeros(self.z_bins,dtype=float)
										
		self._compute_Amatrix()
		self._compute_w_matrix()
		
		#self._plot_against_data()
		
		### Setup log interpolation for the theory cross-correlation predictions
		### Note below need +1 to get to spec-z bin	(element 0 is the photo-z bin)	
		
		self.w_ii_predit_interp = []

		for zbin in range(self.z_bins):
			interp_obj = interp1d(self.theta_array,self.w_ii_predict[zbin+1,0,:],kind='linear',fill_value=0.00,bounds_error=False)	
			self.w_ii_predit_interp.append(interp_obj)

		self.w_interp_cross = lambda theta, bin: self.w_ii_predit_interp[bin](theta)
		
		if(not self.use_schur):		
			self._find_schur_parameters()
			self._find_fisher_matrix()	
			
		if(self.use_schur):
			self.schur_p_temp[:] = 1.0
			self.cross_cor_temp[:,:] = 0.00
			self._find_fisher_schur_limt()
			
		print('computing weights vi_l')
	
		for j in range(self.z_bins):
			j_one = j+1

			self.vi_l[j,:] = (
					self.Nz_spec[j]*self.b_photo[j]*self.b_spec[j]*self.Cl_ii_accurate[j,:]*
					self.schur_p_temp[:]/(self.A_matrix[0,0,:]*self.A_matrix[j_one,j_one,:])
					)
		
# 		fig = pyplot.plot(self.l_int_vec[:],self.vi_l[2,:],color='blue')
# 		fig = pyplot.plot(self.l_int_vec[:],self.vi_l[3,:],color='red')
# 		pyplot.ylabel('Weights vi_l z1')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/l_weights_vi_l.pdf')
		
#		fig = pyplot.plot(self.l_int_vec[:],self.A_matrix[0,2,:],color='blue')
#		pyplot.ylabel('A 02')
#		pyplot.xlabel('l')
#		pyplot.xscale('log')
#		pyplot.show()
		
		#pyplot.savefig('plots/A_matrix_z02.pdf')
		
		self._convert_vi_to_config_space()
					
		if(not self.use_schur):		
		
			print('computing second weights g_{j,k}')
			
			for zbin1 in range(self.z_bins):
				for zbin2 in range(self.z_bins):
					bin1_one,bin2_one = zbin1+1,zbin2+1 
						
					self.g_jk_l[zbin1,zbin2,:] = ( 
							self.vi_l[zbin1,:]*2.0*self.schur_p_temp[:]*
							np.sqrt((self.cross_cor_temp[:,zbin1]**2.*self.cross_cor_temp[:,zbin2]**2.*
							 self.A_matrix[bin1_one,bin1_one,:])/self.A_matrix[bin2_one,bin2_one,:]
							 ))
										
			self._convert_g_jk_to_config_space()
		
		if(not self.use_auto_correlations):
		
			if(self.use_schur):	
				### Fisher Matrix is diagonal in this case, note inverse has already been taken	
								
				return self.vec_weight1[:]*self.fisher_nz_inv_diagonal[:]
				
			if(not self.use_schur):		
				return np.dot(self.fisher_nz_inverse,self.vec_weight1[:] + self.vec_weight2[:])
					
		if(self.use_auto_correlations):
		
			for j in range(self.z_bins):
				j_one = j+1
				self.fj_l[j,:] = self.vi_l[j,:]*(self.A_matrix[0,j_one,:]/self.A_matrix[j_one,j_one,:])
				self.kj_l[j,:] = self.vi_l[j,:]*self.schur_p_temp[:]*(self.A_matrix[0,j_one,:]/self.A_matrix[0,0,:])		
		
			self._convert_fj_to_config_space()
			self._convert_kj_to_config_space()

			self.vec_weights3[:] = self._compute_auto_corrections_vector()
		
			return np.dot(self.fisher_nz_inverse,self.vec_weight1[:] + self.vec_weight2[:] + self.vec_weights3[:])	
						
	def _compute_Amatrix(self):
		""" Calculation for A(l) matrix (see notation from McQuinn and White) """
		
		print('Starting calc for A matrix')
		
		vec = np.zeros([self.z_bins,self.l_int_vec.size])	
		
		for bin_index in range(self.z_bins): 
			vec[bin_index,:] = (self.Nz_photo_guess[bin_index]*self.b_photo[bin_index])**2*self.Cl_ii_accurate[bin_index,:] 
				
		self.A_matrix[0,0,:] = np.sum(vec,axis=0) + self.shot_nose["photo"][0]	
								
		for bin_index in range(self.z_bins): 
		
			self.A_matrix[1+bin_index,1+bin_index,:] = (
				(self.Nz_spec[bin_index]*self.b_spec[bin_index])**2*
				self.Cl_ii_accurate[bin_index,:]) + self.shot_nose["spec"][bin_index] 	
		
			self.A_matrix[0,1+bin_index,:] = self.A_matrix[1+bin_index,0,:] =  (
				(self.Nz_spec[bin_index]*self.b_spec[bin_index]*self.Nz_photo_guess[bin_index]*self.b_photo[bin_index])*
				self.Cl_ii_accurate[bin_index,:]) + self.shot_nose["cross"][bin_index]
				
	def _compute_w_matrix(self):
		""" Calculation for w_i0,w_00 from autocorrelations: scales as N_i(z)b_i(z)."""
				
		vec = np.zeros([self.z_bins,self.theta_array.size])	
		
		for bin_index in range(self.z_bins): 
			vec[bin_index,:] = (self.Nz_photo_guess[bin_index]*self.b_photo[bin_index])**2.0*self.w_ii[bin_index,:] 
			
		self.w_ii_predict[0,0,:] = np.sum(vec,axis=0)
		
		for bin_index in range(self.z_bins): 
					
			self.w_ii_predict[1+bin_index,1+bin_index,:] = (
				(self.Pz_spec[bin_index]*self.b_spec[bin_index])**2.0*self.w_ii[bin_index,:])
		
			self.w_ii_predict[0,1+bin_index,:] = self.w_ii_predict[1+bin_index,0,:] =  (
						(self.b_spec[bin_index]*self.Pz_spec[bin_index])*
						(self.b_photo[bin_index]*self.Pz_photo[bin_index])*self.w_ii[bin_index,:]  
						)
	
	def _plot_against_data(self):
		
		zbins = np.arange(0.025,1.525,0.05)
		
		plt.figure(figsize = (20,20)) # set the figure size to be square
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.1, hspace=0.2, left = 0.1, right = 0.9, bottom = 0.1, top = 0.9) 
		
		axes,row = 4,4
		zbin = 0
		
		for i in range(axes):
			for j in range(row):
				ax = plt.subplot(gs[i, j])
				plt.errorbar(rad_to_arcmin*self.theta_obs_mu,self.w_obs[0,zbin+1,:], yerr=self.error_corr[0,zbin+1,:],fmt='ro',color = 'blue')
				plt.plot(rad_to_arcmin*self.theta_array,self.w_ii_predict[0,1+zbin,:],color = 'red')
				plt.yscale('linear')
				plt.xscale('log')
				
				if(i==3):
					ax.set_xlabel(r'$\theta$ (arcmin)', fontsize=16)
				
				if(j==0):
					ax.set_ylabel(r'$w(\theta)$', fontsize=16)
				
				ax.axhline(color='k', ls='--') 
				print('z bin is',zbin)
				ax.text(5.00, 0.25, '%s$<$z$_{s}$$<$%s'%(zbins[zbin], zbins[zbin+1]), fontsize=14)
				#ax.text(2.00, 0.15, r'spec-z$[z=%s,%s] \times$ photo-z'% (zbins[zbin],zbins[zbin+1]), fontsize=14)
			
				if(i!=3):
					plt.setp(ax.get_xticklabels(), visible=False)
					
				#plt.setp(ax.get_yticklabels(), visible=False)
			
				#plt.setp(ax.set_xticklabels(['1', '10','100']),visible=True)
				
				#if(j==0):
				
				plt.setp(ax.set_yticklabels(['-0.1','-0.05','0', '0.05', '0.1', '0.15', '0.2','0.25', '0.3', '0.35']),visible=True)
				
				#ax.xaxis.tick_top()
		
				plt.tick_params(axis='both', which='major', labelsize='12')
				plt.tick_params(axis='both', which='minor', labelsize='12')
		
				#plt.legend(loc='lower right',prop={'size':7})
				plt.setp(ax.get_xticklabels(), fontsize=16)
				plt.setp(ax.get_yticklabels(), fontsize=16)	
			
				plt.xlim(0.5,100)
				plt.ylim(-0.1, 0.40)
		
				zbin = zbin+1
				
		plt.savefig('W_theta_test-DEEP_lowz.pdf')
		#plt.show()  
		plt.close()
	
		plt.figure(figsize = (20,20)) # set the figure size to be square
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.1, hspace=0.2, left = 0.1, right = 0.9, bottom = 0.1, top = 0.9) 
		
		axes,row = 4,4
		
		zbin = 13
		
		for i in range(axes):
			for j in range(row):
				ax = plt.subplot(gs[i, j])
				plt.errorbar(rad_to_arcmin*self.theta_obs_mu,self.w_obs[0,zbin+1,:], yerr=self.error_corr[0,zbin+1,:],fmt='ro',color = 'blue')
				plt.plot(rad_to_arcmin*self.theta_array,self.w_ii_predict[0,1+zbin,:],color = 'red')
				plt.yscale('linear')
				plt.xscale('log')
				
				if(i==3):
					ax.set_xlabel(r'$\theta$ (arcmin)', fontsize=16)
				
				if(j==0):
					ax.set_ylabel(r'$w(\theta)$', fontsize=16)
				
				ax.axhline(color='k', ls='--') 
				print('z bin is',zbin)
				ax.text(5.00, 0.25, '%s$<$z$_{s}$$<$%s'%(zbins[zbin], zbins[zbin+1]), fontsize=14)
				#ax.text(2.00, 0.15, r'spec-z$[z=%s,%s] \times$ photo-z'% (zbins[zbin],zbins[zbin+1]), fontsize=14)
			
				if(i!=3):
					plt.setp(ax.get_xticklabels(), visible=False)
					
				plt.setp(ax.get_yticklabels(), visible=False)
			
				#plt.setp(ax.set_xticklabels(['1', '10','100']),visible=True)
				
				if(j==0):
					plt.setp(ax.set_yticklabels(['-0.1','-0.05','0', '0.05', '0.1', '0.15', '0.2','0.25', '0.3', '0.35']),visible=True)
				
				#ax.xaxis.tick_top()
		
				plt.tick_params(axis='both', which='major', labelsize='12')
				plt.tick_params(axis='both', which='minor', labelsize='12')
		
				#plt.legend(loc='lower right',prop={'size':7})
				plt.setp(ax.get_xticklabels(), fontsize=16)
				plt.setp(ax.get_yticklabels(), fontsize=16)	
			
				plt.xlim(0.5,100)
				plt.ylim(-0.1, 0.40)
		
				zbin = zbin+1
				
		plt.savefig('W_theta_test-DEEP_highz.pdf')
		#plt.show() 
		plt.close()
	
	def _find_schur_parameters(self):
		"""code to derive cross correlation coefficient r(l) and Schur 
		   parameter S(l); see johnson et al 2016 for the relevant equations"""
		
		for i in range(self.z_bins):
			bin_spec = i+1
			self.cross_cor_temp[:,i] =  (
				self.A_matrix[0,bin_spec,:]/np.sqrt(self.A_matrix[0,0,:]*self.A_matrix[bin_spec,bin_spec,:]))
								
		for l_index,l in enumerate(self.l_int_vec):
			ccor_sum = np.sum(self.cross_cor_temp[l_index][:]**2.)
			if(ccor_sum >= 1.00):
				print('Correcting Cross Corr')
				ccor_sum = 0.99999 ## Temp fix
			self.schur_p_temp[l_index] = 1.0/(1.00 - ccor_sum) 
			
			print('which z bins is causing the problem?',self.cross_cor_temp[l_index][:]**2.)
			print('Ccor_sum is',ccor_sum)
			print('self.schur_p_temp[l_index]',self.schur_p_temp[l_index],l_index)
			
			if(ccor_sum >= 1.00):
				print('found sum of ',ccor_sum)
				exit('problem, divergent terms, shot noise likely to small')
		
		print(' r_i values (l = 400 here)', self.cross_cor_temp[400][i],'for index',i)		
		print(' r_i values (l = 11 here)', self.cross_cor_temp[11][i],'for index',i)	
		print(' S values (l = 400 here)',self.schur_p_temp[400])
		print(' S values (l = 11 here)',self.schur_p_temp[11])	
				
	def _find_fisher_matrix(self):
		"""Fisher Matrix calculation: summation over l,m modes. 
		   Formula derived in McQuinn and White"""
		
		### What l max is needed to find an accurate Fisher matrix? 
		### 10,000 is likely fine? This will change the final errors, check!
		### Restrict range, to not include non-linear Scales. 
		
		l_max_cut,l_min_cut = 1*10**3-2.0,10**1.+ 2.0			
		self.l_int_vec_cut = np.arange(int(l_min_cut) - 1,int(l_max_cut) + 1,dtype='float')
	
		print('check size difference',self.l_int_vec_cut.size,self.l_int_vec.size)
	
		fisher_M_temp = np.zeros([self.z_bins,self.z_bins,self.l_int_vec_cut.size],dtype=float)  
		fisher_M_sum = np.zeros([self.z_bins,self.z_bins],dtype=float)  
		
		for i in range(self.z_bins):
			for j in range(self.z_bins): 
				delta_K = 0.00
				if(i == j): 
					delta_K = 1.00
				#for l_index,l in enumerate(self.l_int_vec):
				for l_index,l in enumerate(self.l_int_vec_cut):					
					deriv_i = self.Nz_spec[i]*self.b_photo[i]*self.b_spec[i]*self.Cl_ii_accurate[i][l_index]
					deriv_j = self.Nz_spec[j]*self.b_photo[j]*self.b_spec[j]*self.Cl_ii_accurate[j][l_index]
				
					sqrt_term = 2.0*self.schur_p_temp[l_index]*np.sqrt(
						((self.cross_cor_temp[l_index][i]**2.0)*(self.cross_cor_temp[l_index][j]**2.0))/
						(self.A_matrix[i+1,i+1,l_index]*self.A_matrix[j+1,j+1,l_index]))
			
					common_factor = self.schur_p_temp[l_index]/self.A_matrix[0,0,l_index]
							
					fisher_M_temp[i,j,l_index] = (common_factor*
						(delta_K/self.A_matrix[i+1,i+1,l_index] + sqrt_term)*deriv_i*deriv_j)
				
		for i in range(self.z_bins):
			for j in range(self.z_bins): 
		
				fisher_M_sum[i,j] = np.sum(
										(2.0*self.l_int_vec_cut[:] + 1.0)*fisher_M_temp[i,j,:])
				
		### Finite sky coverage effect. In Cabre et al. 2007 it was shown, using Gaussian realizations
		### of the a_lm spectra, that errors in configurations space scale as 1 by f sky 
		### This term increases the error based on the available number of harmonic modes
		
		print('Fisher Matrix is',fisher_M_sum[:,:])
		
		fisher_M_sum[:,:] = self.f_sky*fisher_M_sum[:,:]
		
		self.fisher_nz_inverse = np.linalg.inv(fisher_M_sum) 		
		self.errors_fisher[:] = np.sqrt(np.diagonal(self.fisher_nz_inverse))

	def _find_fisher_schur_limt(self):
		"""Simplified Fisher matrix calculation: in this limit the Fisher Matrix is diagonal: 
		   there is no correlation between bins."""
		
		fisher_M_diagonal = np.zeros([self.z_bins,self.l_int_vec.size],dtype=float)  
		fisher_M_sum_diagonal = np.zeros([self.z_bins],dtype=float)  
				
		for i in range(self.z_bins):
							
			deriv_i_vec = self.Nz_spec[i]*self.b_photo[i]*self.b_spec[i]*self.Cl_ii_accurate[i,:]
			factor_vec = 1.0/(self.A_matrix[0,0,:]*self.A_matrix[i+1,i+1,:])
				
			fisher_M_diagonal[i,:] = factor_vec[:]*(deriv_i_vec[:])**2.0
							
		for i in range(self.z_bins):
			
			fisher_M_sum_diagonal[i] = np.sum(
									(2.0*self.l_int_vec[:] + 1.0)*fisher_M_diagonal[i,:])
	
		fisher_M_sum_diagonal[:] = self.f_sky*fisher_M_sum_diagonal[:]
		
		self.fisher_nz_inv_diagonal = 1.0/fisher_M_sum_diagonal[:] 	
		
		self.errors_fisher[:] = np.sqrt(self.fisher_nz_inv_diagonal[:])
		
		for index in range(self.z_bins):
			print('error, index',self.errors_fisher[index],index)
			print('Inverse Fisher, index',self.fisher_nz_inv_diagonal[index],index)
			
	def _convert_vi_to_config_space(self):	
		""" Aim: transform the weights self.vi_l from harmonic space to 
			configuration space this is calculated for each redshift bin. 
			New local theta array is sets integration limits """
																	
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		v_i_theta = np.zeros([self.z_bins,theta_bins],dtype='float')
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]
		
		### _approx is an approximation valid on small scales using spherical bessel functions
		### _full is the standard expression (this only works for vector functions)

		self.vi_theta_kernal_list = []
		
		print('Computing vi weights: starting loop')
		
		for z_bin in range(self.z_bins):
			print('Starting z bin',z_bin)		
			x_array = self.l_int_vec[:]
			y_array = self.vi_l[z_bin,:]
			V_l_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
			v_kernal_approx = lambda l_value,theta: (2.*l_value + 1.)/(4.0*np.pi)*sp.j0(l_value*theta)*V_l_interp(l_value)				
			v_kernal_full = lambda l_value,theta: (2.*l_value+1.)/(4.0*np.pi)*self.Pls_import.Pl(l_value,theta)[:,0]*V_l_interp(l_value)	
			
			for idx,theta in enumerate(theta_array_local):
									
				l_max_bessel_cut = int(j0_limit/theta) 
				l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
				if(theta_cut < theta):
					v_i_theta[z_bin,idx] = np.sum(v_kernal_full(self.l_int_vec[:],theta))
				else:
					v_i_theta[z_bin,idx] = quad(v_kernal_approx,l_min,l_max_bessel_cut,
						args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
			x_array = theta_array_local
			y_array = v_i_theta[z_bin,:]
			
			self.vi_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
			
		### Summation over theta to find weights (see equation 118 in notes). 
		### Note, accuracy of summation limited by delta theta of measurements.
		
		self.vec_weight1 = np.zeros(self.z_bins)
		self.test_weight = np.zeros(self.z_bins)
		
		for zbin in range(self.z_bins):	
			self.vec_weight1[zbin] = sum( 
				self._wv_integrad_summation(x,zbin,self.theta_obs_mu) for x in range(self.theta_obs_mu.size -1))
		
		print('vec_weight1:',self.vec_weight1[:])
	
# 		self.theta_array_new = np.logspace(log_min,log_max,5000)
# 		
# 		for zbin in range(self.z_bins):	
# 			self.test_weight[zbin] = sum( 
# 				self._wv_integrad_summation_v2(x,zbin,self.theta_array_new) for x in range(self.theta_array_new.size -1))
# 
# 		print('Test weights are:',self.test_weight[:])
# 				
# 		### Plot the theta kernal. This shows the scales contributing to the estimator,	
# 		### we use a more accurate theta array for this plot
# 		
# 		theta_array_accurate = np.logspace(log_min,log_max,5000)	
# 		arcmin_to_rad = 0.000290888
# 		rad_to_arcmin = (1.0/arcmin_to_rad)
# 				
# 		plt.figure(figsize = (14,7)) # set the figure size to be square
# 		gs = gridspec.GridSpec(1, 2)
# 		gs.update(wspace=0.3, hspace=0.2, left = 0.1, right = 0.9, bottom = 0.1, top = 0.9) 
# 	
# 		ax = plt.subplot(gs[0, 0])
# 		
# 		weights1_z2 = np.zeros(theta_array_accurate.size)
# 		weights2_z2 = np.zeros(theta_array_accurate.size)
# 		
# 		weights1_z8 = np.zeros(theta_array_accurate.size)
# 		weights2_z8 = np.zeros(theta_array_accurate.size)
# 		
# 		weights1_z2 = self._wv_integrand_old(theta_array_accurate,2)
# 		weights1_z8 = self._wv_integrand_old(theta_array_accurate,8)
# 		
# 		max = np.amax(weights1_z2) 
# 		weights1_z2 = weights1_z2/max
# 		
# 		max = np.amax(weights1_z8) 
# 		weights1_z8 = weights1_z8/max
# 		
# 		weights2_z2 = self._wv_integrand_old_2(theta_array_accurate,2)
# 		weights2_z8 = self._wv_integrand_old_2(theta_array_accurate,8)
# 		
# 		max2 = np.amax(weights2_z2)
# 		weights2_z2 = weights2_z2/max2
# 		
# 		max = np.amax(weights2_z8) 
# 		weights2_z8 = weights2_z8/max
		
# 		fig = pyplot.plot(theta_array_accurate*rad_to_arcmin,weights1_z2,color='red',ls='--')
# 		fig = pyplot.plot(theta_array_accurate*rad_to_arcmin,weights1_z8,color='blue',ls='--')
# 		#pyplot.ylabel('Theta kernal 1')
# 		ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=20,style='italic')
# 		ax.set_ylabel(r'weight', fontsize=20,style='italic')
# 		pyplot.xscale('log')	
# 		plt.xlim(0.01,200)
# 		plt.tick_params(axis='both', which='major', labelsize='12')
# 		plt.tick_params(axis='both', which='minor', labelsize='12')
# 		plt.setp(ax.get_xticklabels(), fontsize=16)
# 		plt.setp(ax.get_yticklabels(), fontsize=16)	
# 		
# 		ax = plt.subplot(gs[0, 1])
# 		fig = pyplot.plot(theta_array_accurate*rad_to_arcmin,weights2_z2,color='red',ls='--')
# 		fig = pyplot.plot(theta_array_accurate*rad_to_arcmin,weights2_z8,color='blue',ls='--')
# 		#plt.ylabel('Theta kernal 2')
# 		plt.xlabel('theta')
# 		plt.xscale('log')
# 		ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=20,style='italic')
# 		ax.set_ylabel(r'weight $\times$ $w_{ps_{i}}(\theta)$', fontsize=20,style='italic')
# 		plt.xlim(0.01,200)
# 		plt.tick_params(axis='both', which='major', labelsize='12')
# 		plt.tick_params(axis='both', which='minor', labelsize='12')
# 		plt.setp(ax.get_xticklabels(), fontsize=16)
# 		plt.setp(ax.get_yticklabels(), fontsize=16)	
# 		
# 		#plt.show()
# 		#plt.savefig('mock_test_cross_corr_new.pdf',bbox_inches='tight',dpi=2000)  
# 		plt.savefig('plots/weights_w_theta.pdf')
# 		exit()	
				
	def _wv_integrad_summation(self,alpha,zbin,theta_vec):
		
		theta_alpha = theta_vec[alpha]
		
		#dtheta = self.theta_obs_bins[alpha+1] - self.theta_obs_bins[alpha]
		
		dtheta = theta_vec[alpha+1]-theta_vec[alpha]
		
		weight =  (8.0*np.pi**2)*dtheta*(
				theta_alpha*self.vi_theta_kernal_list[zbin](theta_alpha))
				
		w_obs_cross = self.w_obs[0,zbin+1,alpha]		
		w_theory_cross = self.w_interp_cross(theta_alpha,zbin)	
		
		### Guess at weight that needs to be added? (Identical to original weights)
		### Perhaps based on assumption on simulation observations?ffffff
		
		#unit_weight = self.Nz_spec[zbin]*self.Nz_photo_guess ### ddddssfff
		
		damping_factor = 10.*3.5*5000.0   #(25,000 n ~ 50, 5000, n = 1000)
		
		unit_weight = (self.Nz_spec[zbin]*self.total_density_pz)/damping_factor
		
		w_diff = unit_weight*(w_obs_cross - w_theory_cross)
										
		return weight*w_diff
	
	def _wv_integrad_summation_v2(self,alpha,zbin,theta_vec):
		
		theta_alpha = theta_vec[alpha]
		dtheta = theta_vec[alpha+1] - theta_vec[alpha]
		
		weight =  (8.0*np.pi**2)*dtheta*(
				theta_alpha*self.vi_theta_kernal_list[zbin](theta_alpha))
				
		w_theory_cross = self.w_interp_cross(theta_alpha,zbin)	

		return weight*(w_theory_cross)
		
	def _wv_integrand_old(self,theta,zbin): 
	
		weight = (8.0*np.pi**2)*theta*self.vi_theta_kernal_list[zbin](theta)
		
		return 	weight
	
	def _wv_integrand_old_2(self,theta,zbin): 
	
		weight = (8.0*np.pi**2)*theta*self.vi_theta_kernal_list[zbin](theta)
		w_theory_cross = self.w_interp_cross(theta,zbin)	
		
		return 	weight*w_theory_cross
					
	def _convert_g_jk_to_config_space(self):
		""" Aim: transform the weights self.g_jk_l from harmonic space to 
			configuration space. This is done at each redshift bin. 
			local_theta_array sets the integration limits"""
			
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		g_jk_theta = np.zeros([self.z_bins,self.z_bins,theta_bins],dtype='float')
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]
		
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)

		self.g_jk_theta_kernal_list = []
		
		### _approx is an approximation valid on small scales using spherical bessel functions 
		### while _full is the standard expression using legendra polynomials
		### This weight term is unfortunately not symmetric plus we need the diagonal components (=slow)
		
		self.index_map = np.arange(self.z_bins**2).reshape(self.z_bins,self.z_bins)
		
		print('Computing G_jk weights: this will take a while')
		
		for bin1 in range(self.z_bins):
			for bin2 in range(self.z_bins):
				
				x_array = self.l_int_vec[:]
				y_array = self.g_jk_l[bin1,bin2,:]
				g_jk_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
				g_kernal_approx = lambda l,theta: (2.*l+1.)/(4.0*np.pi)*(sp.j0(l*theta)*g_jk_interp(l))
				g_kernal_full  = lambda l,theta: (2.*l+1.)/(4.0*np.pi)*(self.Pls_import.Pl(l,theta)[:,0]*g_jk_interp(l))	
					
				for idx,theta in enumerate(theta_array_local):
									
					l_max_bessel_cut = int(j0_limit/theta) 
					l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
					if(theta_cut < theta):
						g_jk_theta[bin1,bin2,idx] = np.sum(g_kernal_full(self.l_int_vec[:],theta))
					else:
						g_jk_theta[bin1,bin2,idx] = quad(g_kernal_approx,l_min,l_max_bessel_cut,
							args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
				x_array = theta_array_local
				y_array = g_jk_theta[bin1,bin2,:]
				
				self.g_jk_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
				
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,2,:],color='blue')
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,3,:],color='red')
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,4,:],color='green')
# 		fig = pyplot.plot(theta_array_local, self.g_jk_theta_kernal_list[self.index_map[1,2]](theta_array_local),color='red')
# 		pyplot.ylabel('Weight g_jk 0')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/theta_weights_g_jk_z1z2.pdf')
		
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,1,:],color='blue')
# 		fig = pyplot.plot(theta_array_local, self.g_jk_theta_kernal_list[self.index_map[1,1]](theta_array_local),color='red')
# 		pyplot.ylabel('Weight g_jk 0')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/theta_weights_g_jk_z1z1.pdf')
				
		self.vec_weight2 = np.zeros(self.z_bins)
		
		for j_index in range(self.z_bins):	
			
			self.vec_weight2[j_index] = sum( 
				self._w_gjk_integrad_summation(x,j_index,self.theta_obs_mu) for x in range(self.theta_obs_mu.size -1))
	
		print('Computed Vector weights 2 are:',self.vec_weight2[:])
								
	def _w_gjk_integrad_summation(self,alpha,j_index,theta_vec):
		
		theta_alpha = theta_vec[alpha]
		d_theta = theta_vec[alpha+1]-theta_vec[alpha]
		
		k_summation = 0 
		
		for k_index in range(self.z_bins):
			
			idx = self.index_map[j_index,k_index]
			g_jk = self.g_jk_theta_kernal_list[idx](theta_alpha)
			
			w_cross_obs = self.w_obs[0,k_index+1,alpha]		
			w_cross_t = self.w_interp_cross(theta_alpha,k_index)			
			
			### AJ updated weights to make consistent with above.
			### Guess at weight that needs to be added? (Identical to original weights)
			### Perhaps based on assumption on simulation observations?
				
			damping_factor = 10.*3.5*5000.0   #(25,000 n ~ 50, 5000, n = 1000)
			
			unit_weight = (self.Nz_spec[k_index]*self.total_density_pz)/damping_factor
		
			w_diff = unit_weight*(w_cross_obs-w_cross_t)
			
			k_summation += g_jk*w_diff
																
		return (8.0*np.pi**2)*d_theta*theta_alpha*k_summation
					
	def _convert_fj_to_config_space(self):	
		""" Aim: transform the weights self.fj_l from harmonic space to 
			configuration space. Very similar to above code"""
																	
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		fj_theta = np.zeros([self.z_bins,theta_bins],dtype='float')
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]
		
		### _approx is an approximation valid on small scales using spherical bessel functions
		### _full is the standard expression (this only works for vector functions)

		self.fj_theta_kernal_list = []
		
		for z_bin in range(self.z_bins):
		
			print('Starting auto terms (fj), bin',z_bin+1)
			x_array = self.l_int_vec[:]
			y_array = self.fj_l[z_bin,:]
			F_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
			kernal_approx = lambda l_value,theta: (2.*l_value + 1.)/(4.0*np.pi)*sp.j0(l_value*theta)*F_interp(l_value)				
			kernal_full = lambda l_value,theta: (2.*l_value+1.)/(4.0*np.pi)*self.Pls_import.Pl(l_value,theta)[:,0]*F_interp(l_value)	
			
			for idx,theta in enumerate(theta_array_local):
									
				l_max_bessel_cut = int(j0_limit/theta) 
				l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
				if(theta_cut < theta):
					fj_theta[z_bin,idx] = np.sum(kernal_full(self.l_int_vec[:],theta))
				else:
					fj_theta[z_bin,idx] = quad(kernal_approx,l_min,l_max_bessel_cut,
						args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
			x_array = theta_array_local
			y_array = fj_theta[z_bin,:]
			
			self.fj_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
			
	def _convert_kj_to_config_space(self):	
		""" Aim: transform the weights self.fj_l from harmonic space to 
			configuration space. Very similar to above code"""
																	
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		kj_theta = np.zeros([self.z_bins,theta_bins],dtype='float')
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]
		
		### _approx is an approximation valid on small scales using spherical bessel functions
		### _full is the standard expression (this only works for vector functions)

		self.kj_theta_kernal_list = []
		
		for z_bin in range(self.z_bins):
		
			print('Starting auto terms (kj), bin',z_bin+1)
			x_array = self.l_int_vec[:]
			y_array = self.kj_l[z_bin,:]
			K_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
			kernal_approx = lambda l_value,theta: (2.*l_value + 1.)/(4.0*np.pi)*sp.j0(l_value*theta)*K_interp(l_value)				
			kernal_full = lambda l_value,theta: (2.*l_value+1.)/(4.0*np.pi)*self.Pls_import.Pl(l_value,theta)[:,0]*K_interp(l_value)	
			
			for idx,theta in enumerate(theta_array_local):
									
				l_max_bessel_cut = int(j0_limit/theta) 
				l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
				if(theta_cut < theta):
					kj_theta[z_bin,idx] = np.sum(kernal_full(self.l_int_vec[:],theta))
				else:
					kj_theta[z_bin,idx] = quad(kernal_approx,l_min,l_max_bessel_cut,
						args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
			x_array = theta_array_local
			y_array = kj_theta[z_bin,:]
			
			self.kj_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
		
		def _compute_auto_corrections_vector(self):
			"""Use all computed weights to compute auto-correlation term"""
	
		term1_temp = np.zeros(self.z_bins) ## Auto for spec-z sample
		term2_temp = np.zeros(self.z_bins) ## Auto for photo-z sample
		term3_temp = np.zeros(self.z_bins) ## Auto for spec-z (summation over k)
		
		### Summation over theta to find weights (see equation 118 in notes). 
		### Note, accuracy of summation limited by delta theta of measurements.
		
		for zbin in range(self.z_bins):	
		
			term1_temp[zbin] = sum( 
				self._wf_integrad_summation(alpha,zbin) for alpha in range(self.theta_obs_mu.size))
				
			term2_temp[zbin] = sum( 
				self._wk_integrad_summation(alpha,zbin) for alpha in range(self.theta_obs_mu.size))	
				
			term3_temp[zbin] = sum( 
				self._w_gjk_integrad_summation_auto(alpha,zbin) for alpha in range(self.theta_obs_mu.size))		
		
		print('term1_temp:',self.self.term1_temp[:])
		print('term2_temp:',self.self.term2_temp[:])
		print('term3_temp:',self.self.term3_temp[:])
		
		### Add code to plot Kernel terms (check which scales are most sensitive for these terms)
		
		return term1_temp[:] + term2_temp[:] + term3_temp[:]
	
	def _wf_integrad_summation(self,alpha,zbin):
	
		theta_alpha = self.theta_obs_mu[alpha]
		dtheta = self.theta_obs_bins[alpha+1] - self.theta_obs_bins[alpha]
	
		weight =  (8.0*np.pi**2)*dtheta*(
		theta_alpha*self.fj_theta_kernal_list[zbin](theta_alpha))
			
		w_ss_obs = self.w_obs[zbin,zbin,alpha]		
		
		return weight*w_ss_obs
		
	def _wk_integrad_summation(self,alpha,zbin):
	
		theta_alpha = self.theta_obs_mu[alpha]
		dtheta = self.theta_obs_bins[alpha+1] - self.theta_obs_bins[alpha]
	
		weight =  (8.0*np.pi**2)*dtheta*(
		theta_alpha*self.kj_theta_kernal_list[zbin](theta_alpha))
			
		w_pp_obs = self.w_obs[0,0,alpha]		
		
		return weight*w_pp_obs

	def _w_gjk_integrad_summation_auto(self,alpha,j_index):
		"""Note factor of 2 difference for weight"""
		
		theta_alpha = self.theta_obs_mu[alpha]
		d_theta = (self.theta_obs_bins[alpha] - self.theta_obs_bins[alpha + 1])
		
		k_summation = 0 
		
		for k_index in range(self.z_bins):
			
			idx = self.index_map[j_index,k_index]
			G_jk = (self.g_jk_theta_kernal_list[idx](theta_alpha))/2.0
			w_auto_ss_obs = self.w_obs[0,0,alpha]		
			
			k_summation += G_jk*w_auto_ss_obs
	
		return (8.0*np.pi**2)*dtheta*theta_alpha*k_summation

	def write(self,outfile,Pz_true,z_mean,errors_fraction,dz):

		col = 4
		write_dat = np.zeros((self.z_bins,col),dtype=float)

		write_dat[:,0] = z_mean[:]
		write_dat[:,1] = Pz_true[:]
		write_dat[:,2] = self.Pz_photo[:]
		write_dat[:,3] = errors_fraction[:] 
				
		np.savetxt(outfile, write_dat,fmt ='%10.5f')
		
		make_plot = False
		
		if(make_plot):
			
			fig = pyplot.figure(1,figsize=(20,20),dpi=400)
			gs1 = gridspec.GridSpec(1,1)
			gs1.update(left=0.15,wspace=0.15,bottom=0.10,hspace=0.15)
	
			ax2 = pyplot.subplot(gs1[0,0])

			handles, labels = ax2.get_legend_handles_labels()
		
			guess, = pyplot.plot(z_mean,self.Nz_photo_guess,color='black',label='$\hat{N}(z)$')
		
			fig = pyplot.errorbar(z_mean, self.Nz_photo_guess, xerr=dz/2, yerr=self.errors_fisher,
					label='$\hat{N}(z)$ (McQuinn and White)',capsize=3,fmt='',ecolor='black')
				
			pyplot.ylabel('$ dN(z)/ d\Omega dz$',fontsize=22)
			pyplot.xlabel('Redshift z',fontsize=22)
			pyplot.legend(handles=[guess],frameon=False,loc = 2,fontsize=22)
			pyplot.xlim(0.05,0.8)
			pyplot.yscale('log')
			pyplot.yticks(fontsize=22)
			pyplot.xticks(fontsize=22)
		
			pyplot.savefig('plots/Nz_estimate.pdf')
				
	def run_test_case(self):	
		""" Aim: To check this code is working by comparing the harmonic space and 
			configuration space estimators: specifically, here we compute the A3 term
			the the result of integration over the weights v_i. This is done for all 
			redshift bins"""
				 
		temp_vec = np.zeros(self.z_bins,dtype=float)
		correction_vec = np.zeros(self.z_bins,dtype=float)
		
		A_term3 = np.zeros(self.z_bins,dtype=float)
		O_l = np.zeros([self.z_bins,self.l_int_vec.size],dtype=float)
				
		self._compute_Amatrix()
		self._compute_w_matrix()
		
		### Setup log interpolation for the theory cross-correlation predictions
		
		self.w_ii_predit_interp = []
		self.w_ii_predit_interp_test = []		
		
		for zbin in range(self.z_bins):
			## Need plus 1 to get to spec-z bin
			interp_obj = interp1d(self.theta_array,self.w_ii_predict[zbin+1,0,:],kind='linear',fill_value=0.00,bounds_error=False)	
			self.w_ii_predit_interp.append(interp_obj)

		self.w_interp_cross = lambda theta, bin: self.w_ii_predit_interp[bin](theta)
							
		if(not self.use_schur):		
			self._find_schur_parameters()
			self.schur_p_temp[:] = 1.0
			self._find_fisher_matrix()	
						
		if(self.use_schur):
			self.schur_p_temp[:] = 1.0
			self.cross_cor_temp[:,:] = 0.00	
			self._find_fisher_schur_limt()
			
		print('computing weights vi_l')
	
		for j in range(self.z_bins):
			j_one = j+1

			self.vi_l[j,:] = (
					self.Nz_spec[j]*self.b_photo[j]*self.b_spec[j]*self.Cl_ii_accurate[j,:]*
					self.schur_p_temp[:]/(self.A_matrix[0,0,:]*self.A_matrix[j_one,j_one,:])
					)
				
# 		fig = pyplot.plot(self.l_int_vec[:],self.vi_l[1,:],color='blue')
# 		pyplot.ylabel('Weights z1')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/l_weights_testcase_z02.pdf')
		
# 		fig = pyplot.plot(self.l_int_vec[:],self.A_matrix[0,2,:],color='blue')
# 		fig = pyplot.plot(self.l_int_vec[:],self.A_matrix[1,1,:],color='green')
# 		fig = pyplot.plot(self.l_int_vec[:],10**12.*self.Cl_ii_accurate[0,:],color='pink')
# 		pyplot.ylabel('A matrix components')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		pyplot.yscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/test_shotnoise.pdf')
				
		self._convert_vi_test_case()
		
		exit()
				
		for j in range(self.z_bins):
			j_one = j+1
			for l_index,l in enumerate(self.l_int_vec):
				A_term3[j] =  A_term3[j] + self.vi_l[j,l_index]*(2*l + 1.0)*(self.A_matrix[0,j_one,l_index]-self.shot_nose["cross"][j])
			
		print('A_3 term are (harmonic space)',A_term3[:])
		
		if(not self.use_schur):		
		
			print('computing second weights g_{j,k}')
			
			for zbin1 in range(self.z_bins):
				for zbin2 in range(self.z_bins):
					bin1_one,bin2_one = zbin1+1,zbin2+1 
						
					self.g_jk_l[zbin1,zbin2,:] = ( 
							self.vi_l[zbin1,:]*2.0*self.schur_p_temp[:]*
							np.sqrt((self.cross_cor_temp[:,zbin1]**2.*self.cross_cor_temp[:,zbin2]**2.*
							 self.A_matrix[bin1_one,bin1_one,:])/self.A_matrix[bin2_one,bin2_one,:]
							 ))
		
			#fig = pyplot.plot(self.l_int_vec[:],self.g_jk_l[1,1,:],color='red')
			#fig = pyplot.plot(self.l_int_vec[:],self.g_jk_l[2,2,:],color='yellow')
			#fig = pyplot.plot(self.l_int_vec[:],self.g_jk_l[1,2,:],color='green')
			#fig = pyplot.plot(self.l_int_vec[:],self.g_jk_l[1,3,:],color='blue')
			#pyplot.ylabel('g_jk matrix components')
			#pyplot.xlabel('l')
			#pyplot.xscale('log')
			#pyplot.show()
			#pyplot.savefig('plots/test_g_jk_matrix.pdf')
		
			self._convert_g_jk_to_config_space_testcase()
		
			test_vec = np.zeros(self.z_bins,dtype=float)
						
			for j in range(self.z_bins):
				j_one = j+1
				for l_index,l in enumerate(self.l_int_vec):
					for k_index in range(self.z_bins):
						test_vec[j] =  test_vec[j] + (
							self.g_jk_l[j,k_index,l_index]*(2*l + 1.0)*(self.A_matrix[0,k_index +1,l_index]-self.shot_nose["cross"][k_index])
							)
		
			print('vector computed in harmonic space is',test_vec[:])	
		
		exit('Finished Test Case')
			
	def _convert_vi_test_case(self):

		""" Test case code: transform the weights self.vi_l from harmonic space to 
			configuration space """
										
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		v_i_theta = np.zeros([self.z_bins,theta_bins],dtype='float')
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]

		### _approx is an approximation valid on small scales using spherical bessel functions
		### _full is the standard expression (this only works for vector functions)
		
		self.vi_theta_kernal_list=[]
		
		for z_bin in range(self.z_bins):
		
			print('Starting bin',z_bin+1)
			x_array = self.l_int_vec[:]
			y_array = self.vi_l[z_bin,:]
			V_l_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
			v_kernal_approx = lambda l_value,theta: (2.*l_value + 1.)/(4.0*np.pi)*sp.j0(l_value*theta)*V_l_interp(l_value)				
			v_kernal_full = lambda l_value,theta: (2.*l_value+1.)/(4.0*np.pi)*self.Pls_import.Pl(l_value,theta)[:,0]*V_l_interp(l_value)	
			
			for idx,theta in enumerate(theta_array_local):
									
				l_max_bessel_cut = int(j0_limit/theta) 
				l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
				if(theta_cut < theta):
					v_i_theta[z_bin,idx] = np.sum(v_kernal_full(self.l_int_vec[:],theta))
				else:
					v_i_theta[z_bin,idx] = quad(v_kernal_approx,l_min,l_max_bessel_cut,
						args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
			x_array = theta_array_local
			y_array = v_i_theta[z_bin,:]
			self.vi_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
		
		#fig = pyplot.plot(theta_array_local, v_i_theta[0,:],color='blue')
		#fig = pyplot.plot(theta_array_local, self.vi_theta_kernal_list[0](theta_array_local),color='red')
		#pyplot.ylabel('Weight vi 0')
		#pyplot.xlabel('l')
		#pyplot.xscale('log')
		#pyplot.show()
		#pyplot.savefig('plots/theta_weights_vi_z0.pdf')
		
		#fig = pyplot.plot(theta_array_local, v_i_theta[1,:],color='blue')
		#fig = pyplot.plot(theta_array_local, self.vi_theta_kernal_list[1](theta_array_local),color='red')
		#pyplot.ylabel('Weight vi 1')
		#pyplot.xlabel('l')
		#pyplot.xscale('log')
		#pyplot.show()
		#pyplot.savefig('plots/theta_weights_vi_z1.pdf')
						
		### New code, uses summation over theta using python generator 
		### expression (see equation 118 in notes)
		
		### Need to use more accurate theta grid here, and check 
		### results are consistent. Seems roughly consistent now.
		
		self.vec_weight1 = np.zeros(self.z_bins)
		
		min_value = np.min(self.theta_array)
		max_value = np.max(self.theta_array)	
		
		log_min_new = np.log10(min_value*self.deg_to_rad)
		log_max_new = np.log10(max_value*self.deg_to_rad)
		self.theta_array_accurate = np.logspace(log_min,log_max,5000)
		
		for zbin in range(self.z_bins):	
		
			self.vec_weight1[zbin] = sum( 
				self._wv_integrad_summation_test_case2(alpha,zbin) for alpha in range(self.theta_array_accurate.size -1))
		
		print('TEST CASE: vec_weight1 using summation, we find',self.vec_weight1[:])
		
		### Old code that uses numerical integration (rather than trap above), check two
		### methods are consistent. Need to re-compute min and max, as logspace does  
		### not have the same endpoints. Interpolating outside this space will 
		### cause errors
					
		check_vec = np.zeros(self.z_bins)
			
		for bin in range(self.z_bins):
			
			check_vec[bin] = quad(self._wv_integrand_old_test_case,min_value,max_value,limit=1000,
					epsrel=0.001,args=(bin,))[0] 
			
		print('TEST CASE: Using full integral one finds',check_vec[:],'should be consistent with above')
				
# 		fig = pyplot.plot(self.theta_array*(1./self.deg_to_rad),self._wv_integrand_old_test_case(self.theta_array,0),color='blue')
# 		pyplot.ylabel('Kernal theta w*V')
# 		pyplot.xlabel('theta')
# 		pyplot.xscale('log')	
# 		#pyplot.show()
# 		pyplot.savefig('plots/theta_kernal_z0.pdf')
		
	def _wv_integrad_summation_test_case(self,alpha,zbin):
		
		theta_alpha = self.theta_array[alpha]
		dtheta = self.theta_array[alpha+1] - self.theta_array[alpha]

		weight =  (8.0*np.pi**2)*dtheta*(
				theta_alpha*self.vi_theta_kernal_list[zbin](theta_alpha))
				
		w_theory_cross = self.w_interp_cross(theta_alpha,zbin)	
		
		return weight*(w_theory_cross)
		
	def _wv_integrad_summation_test_case2(self,alpha,zbin):
		
		theta_alpha = self.theta_array_accurate[alpha]
		dtheta = self.theta_array_accurate[alpha+1] - self.theta_array_accurate[alpha]

		weight =  (8.0*np.pi**2)*dtheta*(
				theta_alpha*self.vi_theta_kernal_list[zbin](theta_alpha))
				
		w_theory_cross = self.w_interp_cross(theta_alpha,zbin)	
		
		return weight*(w_theory_cross)	
		
	def _wv_integrand_old_test_case(self,theta,zbin): 
	
		weight = (8.0*np.pi**2)*theta*self.vi_theta_kernal_list[zbin](theta)
		ans = weight*self.w_interp_cross(theta,zbin)
		
		return 	ans
	
	def _convert_g_jk_to_config_space_testcase(self):
		""" Aim: transform the weights self.g_jk_l from harmonic space to 
			configuration space. This is a piece of code for testing"""
			
		theta_bins = self.theta_properties["num_theta_bins"]
		theta_min = self.theta_properties["theta_min_deg"]
		theta_max = self.theta_properties["theta_max_deg"]
		kernel_bessel_limit = self.theta_properties["kernel_bessel_limit_local"]
		theta_cut = self.theta_properties["theta_approx_cut"]
		
		l_max = self.l_max
		l_min = self.l_min
				
		g_jk_theta = np.zeros([self.z_bins,self.z_bins,theta_bins],dtype='float')
		j0_limit = sp.jn_zeros(0,kernel_bessel_limit)[-1]
		
		log_min = np.log10(theta_min*self.deg_to_rad)
		log_max = np.log10(theta_max*self.deg_to_rad)
		theta_array_local = np.logspace(log_min,log_max,theta_bins)

		self.g_jk_theta_kernal_list = []

		### _approx is an approximation valid on small scales using spherical bessel functions 
		### while _full is the standard expression using legendra polynomials
		### This weight term is unfortunately note symmetric and we need the diagonal components 
		
		self.index_map = np.arange(self.z_bins**2).reshape(self.z_bins,self.z_bins)
		
		for bin1 in range(self.z_bins):
			for bin2 in range(self.z_bins):
				
				print('Starting bins',bin1+1,bin2+1)
				
				x_array = self.l_int_vec[:]
				y_array = self.g_jk_l[bin1,bin2,:]
				g_jk_interp = interp1d(x_array,y_array,kind='linear',fill_value=0.00,bounds_error=False)	
			
				g_kernal_approx = lambda l,theta: (2.*l+1.)/(4.0*np.pi)*(sp.j0(l*theta)*g_jk_interp(l))
				g_kernal_full  = lambda l,theta: (2.*l+1.)/(4.0*np.pi)*(self.Pls_import.Pl(l,theta)[:,0]*g_jk_interp(l))	
					
				for idx,theta in enumerate(theta_array_local):
									
					l_max_bessel_cut = int(j0_limit/theta) 
					l_max_bessel_cut = np.minimum(l_max_bessel_cut,l_max)
		
					if(theta_cut < theta):
						g_jk_theta[bin1,bin2,idx] = np.sum(g_kernal_full(self.l_int_vec[:],theta))
					else:
						g_jk_theta[bin1,bin2,idx] = quad(g_kernal_approx,l_min,l_max_bessel_cut,
							args=(theta,),limit = 1000,epsrel = 0.0001)[0] 
							
				x_array = theta_array_local
				y_array = g_jk_theta[bin1,bin2,:]
				
				self.g_jk_theta_kernal_list.append(InterpolatedUnivariateSpline(x_array,y_array)) 
				
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,2,:],color='blue')
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,3,:],color='red')
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,4,:],color='green')
# 		fig = pyplot.plot(theta_array_local, self.g_jk_theta_kernal_list[self.index_map[1,2]](theta_array_local),color='red')
# 		pyplot.ylabel('Weight g_jk 0')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/theta_weights_g_jk_z1z2.pdf')
# 		
# 		fig = pyplot.plot(theta_array_local, g_jk_theta[1,1,:],color='blue')
# 		fig = pyplot.plot(theta_array_local, self.g_jk_theta_kernal_list[self.index_map[1,1]](theta_array_local),color='red')
# 		pyplot.ylabel('Weight g_jk 0')
# 		pyplot.xlabel('l')
# 		pyplot.xscale('log')
# 		#pyplot.show()
# 		pyplot.savefig('plots/theta_weights_g_jk_z1z1.pdf')
		
		### Summation over theta (see equation 122 in notes)
		### Need to use more accurate theta grid here, and check 
		### results are consistent. Seems roughly consistent now.
		### Hence use self.theta_array_accurate
		
		self.vec_weight2 = np.zeros(self.z_bins)
				
		for j_index in range(self.z_bins):	
			
			self.vec_weight2[j_index] = sum( 
				self._w_gjk_integrad_summation_test(alpha,j_index) for alpha in range(self.theta_array_accurate.size -1))
		
		print('using summation vec2 =',self.vec_weight2[:])
						
		### Old code that uses numerical integration (rather than trap method), check two
		### methods are consistent. Need to re-compute min and max, as logspace does  
		### not use endpoints equal to input values.: problem as interpolating outside this space will 
		### cause errors
			
		min_value = np.min(theta_array_local)
		max_value = np.max(theta_array_local)	
		
		check_vec2 = np.zeros(self.z_bins)
		
		for bin in range(self.z_bins):
			
			check_vec2[bin] = quad(self._w_gjk_integrand_old_test,min_value,max_value,
					limit=1000,epsrel=0.001,args=(bin,))[0] 
			
		print('Using full integral for zbin 1 is',check_vec2[:])
				
# 		fig = pyplot.plot(theta_array_local*(1./self.deg_to_rad),self._w_gjk_integrand_old_test(theta_array_local,1),color='blue')
# 		pyplot.ylabel('Kernal theta w*V')
# 		pyplot.xlabel('theta')
# 		pyplot.xscale('log')	
# 		#pyplot.show()
# 		pyplot.savefig('theta_kernal_zbin2')

	def _w_gjk_integrad_summation_test(self,alpha,j_index):
		
		theta_alpha = self.theta_array_accurate[alpha]
		k_summation = 0 
		
		for k_index in range(self.z_bins):
			
			idx = self.index_map[j_index,k_index]
			g_jk = self.g_jk_theta_kernal_list[idx](theta_alpha)
			w_cross_theory = self.w_interp_cross(theta_alpha,k_index)			
	
			k_summation += g_jk*(w_cross_theory)
				
		d_theta = self.theta_array_accurate[alpha + 1] - self.theta_array_accurate[alpha]
		
		return (8.0*np.pi**2)*d_theta*theta_alpha*k_summation
		
	def _w_gjk_integrand_old_test(self,theta,bin1): 
			
		k_sum = 0 
		for k_index in range(self.z_bins):
			
			idx = self.index_map[bin1,k_index]
			g_jk = self.g_jk_theta_kernal_list[idx](theta)		
			w_cross_theory = self.w_interp_cross(theta,k_index)			
			k_sum += g_jk*w_cross_theory
					
		return (8.0*np.pi**2)*theta*k_sum	

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs	