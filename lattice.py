"""
  Filename: lattice.py
    Author: Daniel Staros
   Summary: Contains functions for defining sites and translation operators on 2D square lattices.
   Imports: numpy, scipy.linalg.block_diag
"""

import numpy as np
from scipy.linalg import block_diag

def site_list(Nx, Ny):
    """ Create list of 2D site coordinates in real space.
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
    """
    site_list = []
    [site_list.append([m,n]) for m in range(0,int(Nx)) for n in range(0,int(Ny))]
    return site_list

def ux(Nx, Ny):
    """ Create position operator matrix along x.
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
    """
    Ntot = Nx * Ny
    if (Nx % int(Nx)) or (Ny % int(Ny)):
        raise TypeError("Nx and Ny must be integers.")
    else:
        ux_mat = np.zeros((Ntot,Ntot))
        sites = site_list(Nx, Ny)

    for i, site_i in enumerate(sites):
        if Nx-1 == site_i[0]:
            ux_mat[i][i] -= 1
        else:
            ux_mat[i][i] += site_i[0]

    full_ux = block_diag(ux_mat,ux_mat)

    return full_ux

def uy(Nx, Ny):
    """ Create position operator matrix along y.
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
    """
    Ntot = Nx * Ny
    if (Nx % int(Nx)) or (Ny % int(Ny)):
        raise TypeError("Nx and Ny must be integers.")
    else:
        uy_mat = np.zeros((Ntot,Ntot))
        sites = site_list(Nx, Ny)

    for i, site_i in enumerate(sites):
        if Ny-1 == site_i[1]:
            uy_mat[i][i] -= 1
        else:
            uy_mat[i][i] += site_i[1]

    full_uy = block_diag(uy_mat,uy_mat)

    return full_uy
