"""
  Filename: hamiltonian.py
    Author: Daniel Staros
   Summary: Contains functions for generating tight-binding Hamiltonians for 2D lattices represented in real-
            space and k-space ("mixed-space"), and evaluating total energy.
   Imports: sys, math, numpy, sympy, math.e, scipy.linalg.block_diag, starlight.lattice, 
            starlight.gutzwiller.sq_a, starlight.fermi.fermi_func
"""

import sys
import math
import numpy as np
import sympy as sp
from math import e
from scipy.linalg import block_diag
from starlight import lattice
from starlight.gutzwiller import sq_a
from starlight.fermi import fermi_func

def dutta_ham_sym(Nx, Ny, W, t, q): # Has gauge A = (0, x*B_z, 0)
    """ Function which generates a *symbolic* real-space tight-binding Hamiltonian given by the model of Dutta et al.
        J. Appl. Phys., 112, 2012. (uses Landau gauge 2: A = (0, x*B_z, 0))
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                 W: Onsite energy broadening (effective "disorder" parameter)
                 t: Hopping parameter
                 q: Integer defining the magnitude of magnetic flux
    """
    Ntot = Nx * Ny
    hamiltonian = sp.zeros(Ntot, Ntot, dtype=complex)
    site_list = lattice.site_list(Nx, Ny)                          # Create list of 2D site coordinates

    # Populate Hamiltonian with hopping terms
    for i in site_list:
        for j in site_list:
            if (j[0] == i[0]) and (j[1] == i[1]):
                hamiltonian[site_list.index(i),site_list.index(j)] += round(np.random.uniform(low = -W/2, high = W/2, size = 1)[0],3)
                        
            if (j[0] != i[0]) and (j[1] == i[1]):
                if j[0] == i[0] + 1:                               # j = [m+1,n], i = [m,n] 
                    hamiltonian[site_list.index(i),site_list.index(j)] += t
                    hamiltonian[site_list.index(j),site_list.index(i)] += t
                if j[0] == i[0] + (Nx - 1):                        # PBC
                    hamiltonian[site_list.index(i),site_list.index(j)] += t
                    hamiltonian[site_list.index(j),site_list.index(i)] += t

            if (j[0] == i[0]) and (j[1] != i[1]):
                m  = i[0]; phi = 1/q; theta = 2*sp.pi*m*phi 
                if j[1] == i[1] + 1:                               # j = [m,n+1], i = [m,n]
                    hamiltonian[site_list.index(i),site_list.index(j)] += t*(exp**(1j*theta))
                    hamiltonian[site_list.index(j),site_list.index(i)] += t*(exp**(-1j*theta))
                if j[1] == i[1] + (Ny - 1):                        # PBC
                    hamiltonian[site_list.index(i),site_list.index(j)] += t*(exp**(-1j*theta))
                    hamiltonian[site_list.index(j),site_list.index(i)] += t*(exp**(1j*theta))
    return hamiltonian

def site_mult_mixed(site, e0):
    return e0*site[1]

def site_rand_mixed(site, W):
    return np.random.uniform(low = -W/2, high = W/2, size = 1)[0]

def calc_e0s(Nx, Ny, func, *args, **kwargs):
    if (not isinstance(Ny, int)) or (not isinstance(Nx, int)):
        raise TypeError("Nx and Ny must be integers.")
    sites = lattice.site_list(Nx, Ny)
    e0s_list = [func(site, *args, **kwargs) for site in sites]

    return e0s_list

def real_ham(Nx, Ny, ta, tb, e_list=False, q=False, gauge=False, occupations=False, d_params=False):
    """ Function which generates a *numerical* real-space tight-binding Hamiltonian.
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            e_list: List of onsite energies
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]))
       occupations: List of average occupations <c_{iσ}^{dagger} c_{iσ}> for each site i (length = Nx * Ny * 2)
          d_params: Array of double occupancy parameters d_i for each site i        (length = Nx * Ny)
    """
    Ntot = Nx * Ny
    if (Nx % int(Nx)) or (Ny % int(Ny)):
        raise TypeError("Nx and Ny must be integers.")
    else:
        hamiltonian = np.zeros((Ntot,Ntot),dtype=complex)          # Define empty Hamiltonian
        site_list = lattice.site_list(Nx, Ny)                      # Define lattice sites
        if q:                                                      # Set default q (magnetic flux parameter) if necessary
            if (gauge == 'L1') or (gauge == False):
                if ((math.gcd(Ny, q) != 1) and (int(Ny) >= int(q))) or (int(Ny) == int(q)): # Ensure q is less than and commensurate with, or equal to Ny
                    phi = 1/q
                    flux_L1 = -1j*2*np.pi*phi
                    flux_L2 = 0
                else:
                    raise ValueError("q must (1) not be 0 or 1, and (2) be commensurate with Nx in the 1st Landau gauge (L1).")
            elif gauge == 'L2':
                if ((math.gcd(Nx, q) != 1) and (int(Nx) >= int(q))) or (int(Nx) == int(q)): # Ensure q is less than and commensurate with, or equal to Nx
                    phi = 1/q
                    flux_L1 = 0
                    flux_L2 = 1j*2*np.pi*phi
                else:
                    raise ValueError("q must (1) not be 0 or 1, and (2) be commensurate with Ny in the 2nd Landau gauge (L2).")
            elif gauge ==  'S':
                print("ERROR: Symmetric gauge is not supported in STARLIGHT. Exiting..."); sys.exit()
            else:
                raise ValueError("Gauge choice invalid. Options are 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]).")
        else:
            flux_L1 = 0
            flux_L2 = 0

        # Add hopping energies/off-diagonal matrix elements
        for i, site_i in enumerate(site_list):
            for j, site_j in enumerate(site_list):
                if (site_j[0] != site_i[0]) and (site_j[1] == site_i[1]):
                    m, n = site_i
                    if site_j[0] == site_i[0] + 1:             # Hopping in x-direction between site_i = [m,n] and site_j = [m+1,n]
                        hamiltonian[i][j] += -ta*(e**(flux_L1*n))
                        hamiltonian[j][i] += -ta*np.conj((e**(flux_L1*n)))
                    if site_j[0] == site_i[0] + (Nx - 1):      # Periodic boundary hopping between site_i = [m,n] and site_j = [m+(Nx-1),n]
                        hamiltonian[i][j] += -ta*np.conj((e**(flux_L1*n)))
                        hamiltonian[j][i] += -ta*(e**(flux_L1*n))
                elif (site_j[0] == site_i[0]) and (site_j[1] != site_i[1]):
                    m, n = site_i
                    if site_j[1] == site_i[1] + 1:             # Hopping in y-direction between site_i = [m,n] and site_j = [m,n+1]
                        hamiltonian[i][j] += -tb*(e**(flux_L2*m))
                        hamiltonian[j][i] += -tb*np.conj((e**(flux_L2*m)))
                    if site_j[1] == site_i[1] + (Ny - 1):      # Periodic boundary hopping between site_i = [m,n] and site_j = [m,n+(Ny-1)]
                        hamiltonian[i][j] += -tb*np.conj((e**(flux_L2*m)))
                        hamiltonian[j][i] += -tb*(e**(flux_L2*m))

        full_ham = block_diag(hamiltonian,hamiltonian)

        # Add onsite energies
        if e_list:
            for i in range(len(full_ham)):
                full_ham[i][i] += e_list[i]

        # Incorporate Gutzwiller renormalization if appropriate
        if occupations and d_params.any():
            if len(site_list) == len(d_params):                      # Fully spatially unrestricted
                pass
            elif len(d_params) == 1:                                 # Isotropic
                d_params = np.repeat(d_params,len(site_list))
                occupation_list = np.repeat(occupations,len(site_list)*2)
                occupations = occupation_list
            else:
                raise ValueError("d_params must be of length 1 (isotropic), or len(number of lattice sites) (spatially unrestricted).")

            ham_s      = full_ham[Nx*Ny:, Nx*Ny:]
            ham_sbar   = full_ham[:Nx*Ny, :Nx*Ny]
            ns_list    = occupations[:Nx*Ny]
            nsbar_list = occupations[Nx*Ny:]
            # Renormalize hopping terms
            for i, site_i in enumerate(site_list):
                for j, site_j in enumerate(site_list):
                    sqa_i_s    = sq_a(ns_list[i], nsbar_list[i], d_params[i])
                    sqa_j_s    = sq_a(ns_list[j], nsbar_list[j], d_params[j])
                    sqa_i_sbar = sq_a(nsbar_list[i], ns_list[i], d_params[i])
                    sqa_j_sbar = sq_a(nsbar_list[j], ns_list[j], d_params[j])
                    ham_s[i][j]    *= sqa_i_s*sqa_j_s
                    ham_sbar[i][j] *= sqa_i_sbar*sqa_j_sbar

            full_ham = block_diag(ham_s, ham_sbar)

    return full_ham

def ham_plaq(ta, tb, kx, ky, p, q, occupations=False, d_params=False, e_list=False):
    """ Function which generates a *numerical* plaquette Hamiltonian in k-space
        PARAMETERS
                ta: Quantifies hopping energy in x
                tb: Quantifies hopping energy in y
                kx: x-component of k-point at which to construct the Hamiltonian
                ky: y-component of k-point at which to construct the Hamiltonian
                 p: An integer defining magnetic flux (must be relatively prime with q)
                 q: An integer defining magnetic flux (must be relatively prime with p)
       occupations: List of site occupations defining Gutzwiller parameters (length = q)
          d_params: Array of double occupancy parameters defining Gutzwiller parameters (length = q)
            e_list: List of site energies
    """
    # Construct generic Hamiltonian template (of size q by q)
    ham = np.zeros((q, q), dtype=complex)
    for j in range(0,q):
        phi = p/q
        A = 2*np.pi*phi*(j)
        vj = -2*ta*np.cos(kx + A)
        ham[j][j] += vj
        #ham[i][i] += e_list[i]
        if j in range(0,q-1):
            ham[j+1][j] = -tb
            ham[j][j+1] = -tb
    # Periodic boundary hops
    ham[0][q-1] = -tb*np.exp( 1j*q*ky)
    ham[q-1][0] = -tb*np.exp(-1j*q*ky)

    ham_s  = ham.copy()
    ham_sb = ham.copy()

    # Add onsite energies if provided (must be of length 2q)
    if e_list:
        ham_s  += np.diag(e_list[:q])
        ham_sb += np.diag(e_list[q:])

    if occupations and d_params.any():
        for i in range(0,q):
            for j in range(0,q):
                # Calculate Gutzwiller renormalization parameters for s and sbar channels
                sqa_i_s  = sq_a(occupations[i], occupations[i+q], d_params[i])
                sqa_i_sb = sq_a(occupations[i+q], occupations[i], d_params[i])
                sqa_j_s  = sq_a(occupations[j], occupations[j+q], d_params[j])
                sqa_j_sb = sq_a(occupations[j+q], occupations[j], d_params[j])
                ham_s[i][j]  *= sqa_i_s*sqa_j_s
                ham_sb[i][j] *= sqa_i_sb*sqa_j_sb

    return ham_s, ham_sb

def expectation_muVT(eigvals, mu, T=False):
    """ Function which calculates the grand canonical expectation value of a numerical TB Hamiltonian.
        PARAMETERS
           eigvals: Eigenvalues obtained by diagonalizing a model Hamiltonian
                mu: Chemical potential
                 T: Temperature in units of Kelvin (default corresponds to kBT = 0.01)
    """
    expectation = 0
    for eigval in eigvals:
        summand = fermi_func(eigval, mu, T=T)*eigval
        expectation += summand
    return expectation

