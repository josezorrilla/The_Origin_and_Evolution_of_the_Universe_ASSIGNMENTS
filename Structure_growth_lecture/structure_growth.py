# To display images
from IPython.display import Image
import numpy as np
from ipywidgets import interactive
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

import astropy.units as u           # we will use this library to handle units
from astropy.constants import c     # speeed of light
from scipy.integrate import quad    # only needed if we use quad function for integral

class FLRWcosmo:
    '''Class for cosmological model representing a uniform and isotropic universe described by GR'''
    
    def __init__(self, Om_0, Or_0, Ol_0, H_0=72*u.km/u.s/u.Mpc):
        # Since the uncertainty on the Hubble constant is not too big, we can 
        # add a default value for it
        # Feel free to modify the class and add default values for your favourite model
        self.Om_0 = Om_0
        self.Or_0 = Or_0
        self.Ol_0 = Ol_0
        self.Ok_0 = 1 - self.Om_0 - self.Or_0 - self.Ol_0
        self.H_0  = H_0
        
        # Derived parameters that will be used to normalize distances
#        from astropy.constants import c
        self.t_H = (1 / self.H_0).to(u.Gyr).value     # Hubble time in Gyr
        self.D_H = (c / self.H_0).to(u.Mpc).value   # Hubble radius in Mpc
        
    # Scaling of enery densities per component at scale factor a
    def Om(self, a):
        '''Non-relativistic matter energy density in units of critical density at a=1'''       
        omega_m = self.Om_0 / a**3 
        return omega_m
    
    def Or(self, a):
        '''Radiation energy density in units of critical density at a=1'''
        omega_r = self.Or_0 / a**4
        return omega_r
    
    def Ol(self, a):
        '''Cosmological constant energy density in unist of critical density at a=1'''
        omega_l = self.Ol_0 
        return omega_l
    
    def Ok(self, a):
        '''Curvature energy density in units of critical density at a=1'''
        omega_k = self.Ok_0 / a**2
        return omega_k
    
    def Omega(self, a):
        '''Total energy density in units of critical density at a=1'''
        omega = self.Om(a) + self.Or(a) + self.Ol(a) + self.Ok(a)
        return omega
    
    def E(self, a):
        '''Square root of energy density, enters Friedmann's equation'''
        
        E = self.Omega(a)**0.5
        
        return E
    
    # Ages
    def age(self, a):
        '''Age of the universe, in Gyr by integrating the Friedman's equation'''
        
        # implement the integral for chi
        def integrand(x):
            '''Computes the integrand, for chi given scale factor x'''
            
            integ = 1 / x / self.E(x)
            
            return integ
        
        # perform the integral
        
        chi = quad(integrand, 0, a)[0]
        
        # scale with Hubble time, bringing back units in Gyr
        
        age = self.t_H * chi
        
        return age
    
    def look_back_time(self, a):
        '''Returns lookback time to an object at scale factor a'''
        
        t_0 = self.age(1)
        t_e = self.age(a)
        
        return t_0 - t_e
    
    # Distances
    # Implements formulae from Hogg 2001: https://arxiv.org/pdf/astro-ph/9905116.pdf
    def comoving_radial_distance(self, a):
        '''Returns comoving radial distance to object at scale factor a in Mpc'''
        
        D_C = self.D_H * quad(lambda x: 1 / x**2 / self.E(x), a, 1)[0]
        
        return D_C
    
    def proper_radial_distance(self, a):
        '''Returns physical (proper) radial distance to object at scale factor a in Mpc'''
        
        D_P = - self.D_H * quad(lambda x : 1 / x / self.E(x), a, 1)[0]
        
        return D_P
    
    def comoving_transverse_distance(self, a):
        '''Returns transverse comoving distance to objec at scale factor a in Mpc'''
        
        if self.Ok_0 > 0:
            
            D_M = self.D_H  / self.Ok_0**0.5 * np.sinh(self.Ok_0**0.5 * self.comoving_radial_distance(a) / self.D_H)
        
        elif self.Ok_0 == 0:
            
            D_M = self.comoving_radial_distance(a)
       
        else:
            
            D_M = self.D_H / (-self.Ok_0)**0.5 * np.sin((-self.Ok_0)**0.5 * self.comoving_radial_distance(a) / self.D_H)
            
        return D_M
    
    def luminosity_distance(self, a):
        '''Returns luminosity distance to object at scale factor a in Mpc'''
        
        D_L = self.comoving_transverse_distance(a) / a
        
        return D_L
    
    def angular_diameter_distance(self, a_1, a_2=None):
        '''Returns angular diameter distance between two objects at scale factors a_1 and a_2,
        if only one scale factor is given, returns angular diameter distance between object and observer
        at a=1, in Mpc'''
        
        if a_2 is None:
            
            D_A = self.comoving_transverse_distance(a_1) * a_1
            
        else:
            
            D_M1 = self.comoving_radial_distance(a_1)
            D_M2 = self.comoving_radial_distance(a_2)
            D_A = a_2 * (D_M2 * (1 + self.Ok_0 * (D_M1 / self.D_H)**2)**0.5 - 
                         D_M1 * (1 + self.Ok_0 * (D_M2 / self.D_H)**2)**0.5)
            
        return D_A   
    
    def comoving_volume_element(self, a):
        '''Returns comoving volume element dV_c / dOmega / da, that is, element comoving volume per 
        unit scale factor and per unit solid angle'''
        
        dv_da_domega = self.D_H * self.angular_diameter_distance(a)**2 / self.E(a) / a**4
        
        return dv_da_domega