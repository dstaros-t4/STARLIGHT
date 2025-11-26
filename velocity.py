"""
  Filename: velocity.py
    Author: Daniel Staros
   Summary: Contains functions for building the velocity operators needed to calculate the transverse and longitudinal conductivity.
   Imports: sys, numpy, math.e, math.gcd, starlight.lattice, starlight.gutzwiller.sq_a
"""

import sys
import math
import numpy as np
from math import e, gcd
from scipy.linalg import block_diag
from starlight import lattice
from starlight.gutzwiller import sq_a


def vx(Nx, Ny, ta, tb, e_list=False, q=False, gauge=False, occupations=False, d_params=False):
    """ Function which generates the real-space velocity operator in x
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            e_list: List of site energies per site/row
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]))
       occupations: List of average occupations <c_{iσ}^{dagger} c_{iσ}> for each site i (length = Nx * Ny * 2)
          d_params: Array of double occupancy parameters d_i for each site i        (length = Nx * Ny)
    """
    h = 1; hbar = h/(2*np.pi)
    Ntot = Nx * Ny
    if (Nx % int(Nx)) or (Ny % int(Ny)):
        raise TypeError("Nx and Ny must be integers.")
    else:
        vx_mat = np.zeros((Ntot,Ntot),dtype=complex)          # Define empty Hamiltonian
        site_list = lattice.site_list(Nx, Ny)                      # Define lattice sites
        if q:                                                      # Set default q (magnetic flux parameter) if necessary
            if (gauge == 'L1') or (gauge == False):
                if ((math.gcd(Ny, q) != 1) and (int(Ny) >= int(q))) or (int(Ny) == int(q)): # Ensure q is less than and commensurate with, or equal to Ny
                    phi = 1/q
                    flux_L1 = -1j*2*np.pi*phi
                    flux_L2 = 0
                else:
                    raise ValueError("q must be commensurate with, but not equal to, Nx in the 1st Landau gauge (L1).")
            elif gauge == 'L2':
                if ((math.gcd(Nx, q) != 1) and (int(Nx) >= int(q))) or (int(Nx) == int(q)): # Ensure q is less than and commensurate with, or equal to Nx
                    phi = 1/q
                    flux_L1 = 0
                    flux_L2 = 1j*2*np.pi*phi
                else:
                    raise ValueError("q must be commensurate with, but not equal to, Ny in the 2nd Landau gauge (L2).")
            elif gauge ==  'S':
                print("ERROR: Symmetric gauge is not supported in STARLIGHT. Exiting..."); sys.exit()
            else:
                raise ValueError("Gauge choice invalid. Options are 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]).")

        else:
            flux_L1 = 0

        # Add diagonal matrix elements to Hamiltonian
        for i, site_i in enumerate(site_list):

            # Add hopping energies/off-diagonal matrix elements
            for j, site_j in enumerate(site_list):
                if (site_j[0] != site_i[0]) and (site_j[1] == site_i[1]):
                    n = site_i[1]
                    if site_j[0] == site_i[0] + 1:             # Hopping in x-direction between site_i = [m,n] and site_j = [m+1,n]
                        vx_mat[i][j] +=  ta*(e**(flux_L1*n))
                        vx_mat[j][i] += -ta*np.conj((e**(flux_L1*n)))
                    if site_j[0] == site_i[0] + (Nx - 1):      # Periodic boundary hopping between site_i = [m,n] and site_j = [m+(Nx-1),n]
                        vx_mat[i][j] += -ta*np.conj((e**(flux_L1*n)))
                        vx_mat[j][i] +=  ta*(e**(flux_L1*n))

        full_vx = block_diag(vx_mat,vx_mat)

        # Incorporate Gutzwiller renormalization if appropriate
        if occupations and d_params.any():
            if len(site_list) == len(d_params):                      # Fully spatially unrestricted
                pass
            elif len(d_params) == 1:                                 # Isotropic
                d_params = np.repeat(d_params,len(site_list))
            else:
                raise ValueError("d_params must be of length 1 (isotropic), or len(number of lattice sites) (spatially unrestricted).")

            vx_s       = full_vx[Nx*Ny:, Nx*Ny:]
            vx_sbar    = full_vx[:Nx*Ny, :Nx*Ny]
            ns_list    = occupations[:Nx*Ny]
            nsbar_list = occupations[Nx*Ny:]
            # Renormalize hopping terms
            for i, site_i in enumerate(site_list):
                for j, site_j in enumerate(site_list):
                    if not i == j:
                        sqa_i_s    = sq_a(ns_list[i], nsbar_list[i], d_params[i])
                        sqa_j_s    = sq_a(ns_list[j], nsbar_list[j], d_params[j])
                        sqa_i_sbar = sq_a(nsbar_list[i], ns_list[i], d_params[i])
                        sqa_j_sbar = sq_a(nsbar_list[j], ns_list[j], d_params[j])
                        vx_s[i][j]    *= sqa_i_s*sqa_j_s
                        vx_sbar[i][j] *= sqa_i_sbar*sqa_j_sbar

            full_vx = block_diag(vx_s, vx_sbar)

    vx_op = (1j/hbar)*full_vx
    return vx_op


def vy(Nx, Ny, ta, tb, e_list=False, q=False, gauge=False, occupations=False, d_params=False):
    """ Function which generates the real-space velocity operator in y
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            e_list: List of site energies per site/row
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]))
       occupations: List of average occupations <c_{iσ}^{dagger} c_{iσ}> for each site i (length = Nx * Ny * 2)
          d_params: Array of double occupancy parameters d_i for each site i        (length = Nx * Ny)
    """
    h = 1; hbar = h/(2*np.pi)
    Ntot = Nx * Ny
    if (Nx % int(Nx)) or (Ny % int(Ny)):
        raise TypeError("Nx and Ny must be integers.")
    else:
        vy_mat = np.zeros((Ntot,Ntot),dtype=complex)          # Define empty Hamiltonian
        site_list = lattice.site_list(Nx, Ny)                      # Define lattice sites
        if q:                                                      # Set default q (magnetic flux parameter) if necessary
            if (gauge == 'L1') or (gauge == False):
                if ((math.gcd(Ny, q) != 1) and (int(Ny) >= int(q))) or (int(Ny) == int(q)): # Ensure q is less than and commensurate with, or equal to Ny
                    phi = 1/q
                    flux_L2 = 0
                else:
                    raise ValueError("q must be commensurate with, but not greater than Ny in the 1st Landau gauge (L1).")
            elif gauge == 'L2':
                if ((math.gcd(Nx, q) != 1) and (int(Nx) >= int(q))) or (int(Nx) == int(q)): # Ensure q is less than and commensurate with, or equal to Nx
                    phi = 1/q
                    flux_L2 = 1j*2*np.pi*phi
                else:
                    raise ValueError("q must be commensurate with, but not greater than, Nx in the 2nd Landau gauge (L2).")
            elif gauge ==  'S':
                print("ERROR: Symmetric gauge is not supported in STARLIGHT. Exiting..."); sys.exit()
            else:
                raise ValueError("Gauge choice invalid. Options are 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]).")

        else:
            flux_L2 = 0

        # Add hopping energies/off-diagonal matrix elements
        for i, site_i in enumerate(site_list):
            for j, site_j in enumerate(site_list):
                if (site_j[0] == site_i[0]) and (site_j[1] != site_i[1]):
                    m = site_i[0]
                    if site_j[1] == site_i[1] + 1:             # Hopping in x-direction between site_i = [m,n] and site_j = [m+1,n]
                        vy_mat[i][j] +=  tb*(e**(flux_L2*m))
                        vy_mat[j][i] += -tb*np.conj((e**(flux_L2*m)))
                    if site_j[1] == site_i[1] + (Ny - 1):      # Periodic boundary hopping between site_i = [m,n] and site_j = [m+(Nx-1),n]
                        vy_mat[i][j] += -tb*np.conj((e**(flux_L2*m)))
                        vy_mat[j][i] +=  tb*(e**(flux_L2*m))

        full_vy = block_diag(vy_mat,vy_mat)

        # Incorporate Gutzwiller renormalization if appropriate
        if occupations and d_params.any():
            if len(site_list) == len(d_params):                      # Fully spatially unrestricted
                pass
            elif len(d_params) == 1:                                 # Isotropic
                d_params = np.repeat(d_params,len(site_list))
            else:
                raise ValueError("d_params must be of length 1 (isotropic), or len(number of lattice sites) (spatially unrestricted).")

            vy_s       = full_vy[Nx*Ny:, Nx*Ny:]
            vy_sbar    = full_vy[:Nx*Ny, :Nx*Ny]
            ns_list    = occupations[:Nx*Ny]
            nsbar_list = occupations[Nx*Ny:]
            # Renormalize hopping terms
            for i, site_i in enumerate(site_list):
                for j, site_j in enumerate(site_list):
                    if not i == j:
                        sqa_i_s    = sq_a(ns_list[i], nsbar_list[i], d_params[i])
                        sqa_j_s    = sq_a(ns_list[j], nsbar_list[j], d_params[j])
                        sqa_i_sbar = sq_a(nsbar_list[i], ns_list[i], d_params[i])
                        sqa_j_sbar = sq_a(nsbar_list[j], ns_list[j], d_params[j])
                        vy_s[i][j]    *= sqa_i_s*sqa_j_s
                        vy_sbar[i][j] *= sqa_i_sbar*sqa_j_sbar

            full_vy = block_diag(vy_s, vy_sbar)

    vy_op = (1j/hbar)*full_vy
    return vy_op
