"""
  Filename: fermi.py
    Author: Daniel Staros
   Summary: Contains functions for evaluating the Fermi-Dirac function, constraining particle number via Fermi-Dirac
            statistics, and calculating the chemical potential (mu) of tight-binding Hamiltonians at finite temperature.
   Imports: math, numpy,  starlight.lattice, scipy.optimize.brentq
"""

import math
import numpy as np
from math import e
from starlight import lattice
from scipy.optimize import brentq

def fermi_func(ener, mu, T=False):
    """ Evaluates the Fermi-Dirac distribution function.
      --parameters--
        ener: Energy at which to evaluate the Fermi function
          mu: Chemical potential
           T: Temperature in units of t (default corresponds to kBT = 0.01 t)
    """
    kB = 1 # STARLIGHT uses natural units; this means that T will have units of t
    if T is False: # Use same kBT as Dutta, P. et al. J. Appl. Phys. 112, 2012
        kBT = 0.01
        if (ener-mu > 7):  # Prevent overflow errors; this does not change the answer
            return 0
        else:    
            return (1 + e**((ener - mu)/(kBT)))**(-1)
    if T < 0:
        raise ValueError("Temperature must be non-negative")
    if T == 0:
        if ener < mu:
            return 1
        elif ener > mu:
            return 0
        else:  # ener == mu
            return 1 / 2
    else:
        if (ener-mu > 7):
            return 0
        else:
            return 1 / (e**((ener - mu) / (kB*T)) + 1)

def fermi_sum_const(mu, Nx, Ny, eigvals, filling, T=False):
    """ Particle number constraint via the sum of Fermi-Dirac function evaluations.
      --parameters--
                mu: Chemical potential
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
           eigvals: Eigenvalues obtained by diagonalizing a model Hamiltonian
           filling: Filling factor which can range from 0.0 to 1.0 (e.g 0.5 = half filling)
                 T: Temperature in units of t (default corresponds to kBT = 0.01 t)
    """
    fermi_sum = 0
    fermi_sum += np.sum([fermi_func(eigval, mu, T=T) for eigval in eigvals])
    Ne_tot = 2*Nx*Ny*filling # Factor of 2 accounts for spin degeneracy
    return fermi_sum-Ne_tot

def mu_count(eigvals, filling):
    """ Calculates the Fermi level by counting states. Valid alone for symmetric densities of 
        states (DOS), but must be succeeded by mu_root for asymmetric DOS.
      --parameters--
           eigvals: Eigenvalues obtained by diagonalizing a tight-binding Hamiltonian
           filling: Filling factor; allowed range of (0.0 to 1.0] (e.g 0.5 = half filling)
    """
    num_states  = len(eigvals)
    state_HO    = math.floor(filling*num_states) # Highest occupied "HO" state
    mu_count  = eigvals[state_HO-1]
    return mu_count

def mu_root(fermi_guess, Nx, Ny, eigvals, filling, T=False):
    """ Calculates the Fermi level by constraining particle number. Valid for symmetric and
        asymmetric DOS, but should match starlight.fermi.mu_count() for symmetric DOS.
      --parameters--
           fermi_guess: Energy at which to evaluate the Fermi function
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
           eigvals: Eigenvalues obtained by diagonalizing a tight-binding Hamiltonian
           filling: Filling factor; allowed range of (0.0 to 1.0] (e.g 0.5 = half filling)
                 T: Temperature in units of t (default corresponds to kBT = 0.01 t)
    """
    mu_low  = fermi_guess - 1; mu_high = fermi_guess + 1
    mu_root = brentq(fermi_sum_const, mu_low, mu_high, args=(Nx, Ny, eigvals, filling, T))
    return mu_root

def mu_calc(Nx, Ny, eigvals, filling, T=False):
    """ Computes the Fermi level by (1) establishing an initial guess based on state counting (starlight.
        fermi.mu_count(), and then (2) correcting based on particle number conservation (starlight.fermi.
        mu_root()).
      --parameters--
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
           eigvals: Eigenvalues obtained by diagonalizing a model Hamiltonian
           filling: Filling factor which can range from 0.0 to 1.0 (e.g 0.5 = half filling)
                 T: Temperature in units of t (default corresponds to kBT = 0.01 t)
    """
    mu_init = mu_count(eigvals, filling)
    fermi_level = mu_root(mu_init, Nx, Ny, eigvals, filling, T=T)
    return fermi_level
