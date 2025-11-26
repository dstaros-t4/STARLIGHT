"""
  Filename: plotting.py
    Author: Daniel Staros
   Summary: Contains functions which automate the plotting of matrix elements, energy spectra, and other observables.
   Imports: math, inspect, numpy, matplotlib, matplotlib.pyplot
"""

import math
import inspect
import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(matrix, figsize, cmap=False, components=False):
    """ Function which generates a heat map of complex matrix elements.
        PARAMETERS
            matrix: Matrix to plot (hamiltonian, velocity operators vx/vy, etc.)
           figsize: Figure size in format 'figsize=(4,6)'
              cmap: User-defined color map option. Default is viridis
        components: Components of the matrix. 'absolute' is default, other options are 'real' or 'imag'
           xlimits: Column range of plot
           ylimits: Row range of plot
    """

    # Format of figure size: figsize = (W,L)
    plt.figure(figsize=figsize)

    if not cmap:
        cmap = 'viridis'

    # Obtain the name of the matrix for plot title
    frame = inspect.currentframe()
    calling_locals = frame.f_back.f_locals
    matrix_name = [name for name, val in calling_locals.items() if val is matrix]
    if matrix_name:
        matrix_name = matrix_name[0]
    else: # Default
        matrix_name = 'matrix'

    if not components:
        plt.imshow(np.abs(matrix), cmap='viridis', aspect='auto')
        plt.title(f'Absolute values of {matrix_name}')
    elif components=='real':
        plt.imshow(np.real(matrix), cmap='viridis', aspect='auto')
        plt.title(f'Real components of {matrix_name}')
    elif components=='imag':
        plt.imshow(np.imag(matrix), cmap='viridis', aspect='auto')
        plt.title(f'Imaginary components of {matrix_name}')

    else:
        raise ValueError("Options for 'components' are 'real' or 'imaginary'.")
        
    plt.xlabel('Annihilation operator index')
    plt.ylabel('Creation operator index')
    plt.colorbar(label='Magnitude')
    plt.show()

def plot_eigvals(eigvals, x_min=False, x_max=False, linecolor=False, linestyle=False, linewidth=False):
    """ Function which plots eigenvalues as vertical lines.
        PARAMETERS
           eigvals: Eigenvalues for a real- or mixed-space model
    """
    if not x_min and not x_max:
        print("Using default x_min and x_max. Define both to plot custom range")
        x_min = min(eigvals)-0.01
        x_max = max(eigvals)+0.01
    else:
        x_min = x_min
        x_max = x_max
    fig, ax = plt.subplots()

    if not linecolor:
        linecolor = 'blue'
    if not linestyle:
        linestyle = '-'
    if not linewidth:
        linewidth = 0.5

    for x in eigvals:
        ax.axvline(x=x, color=linecolor, linestyle=linestyle, linewidth=linewidth)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Energy ($t$)')
    ax.set_yticks([])
    plt.grid(False)
    plt.show()

def plot_params(parameters, Nx, Ny, vmin=False, vmax=False):
    """ Function which plots k-space-optimized Gutzwiller parameters in real-space.
        PARAMETERS
        parameters: List of k-space-optimized parameters (length q)
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction (must be greater than and commensurate with q)
    """

    q          = len(parameters)
    if Ny % q:
        raise ValueError('Ny must be larger than, and commensurate with the number of parameters.')
    param_tile = parameters*int(Ny/q)
    param_mat  = np.array([param_tile for _ in range(Nx)]).T
    if vmin and vmax:
        plt.matshow(param_mat, vmin=vmin, vmax=vmax)
    else:
        plt.matshow(param_mat)
    plt.colorbar(shrink=0.8)
    plt.show()

def plot_conductivity(mu_values, sigma_values, direction, xlim=False, ylim=False, figsize=False, linecolor=False, linestyle=False, linewidth=False):
    """ Function which plots eigenvalues as vertical lines.
        PARAMETERS
         mu_values: List of chemical potentials (x-data)
      sigma_values: List of conductivity evaluations (y-data)
         direction: For plotting labels; conductivity direction. Options are 'xx', 'yy', 'xy'
    """
    if not figsize:
        plt.figure(figsize=(7, 4.5))
    else:
        plt.figure(figsize=figsize)

    if not linecolor:
        linecolor = 'darkviolet'
    if not linestyle:
        linestyle = '-'
    if not linewidth:
        linewidth = 2
    if not xlim:
        xlim = min(mu_values), max(mu_values)
    if not ylim:
        ylim = min(sigma_values)-0.02, max(sigma_values)+0.02

    plt.plot(mu_values, sigma_values, linestyle=linestyle, color=linecolor, linewidth=linewidth)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('$\\mu$ (t)', fontsize=14); plt.ylabel(f'$\\sigma_{{{direction}}}$', fontsize=14); plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_bands_kx(kx_values, eigvals_array, kx_range=False, ener_range=False, hline=False, figsize=False, display=False, save_png=False):
    """ Function which plots band structure along kx.
        PARAMETERS
         kx_values: List of kx points at which energies were evaluated.
     eigvals_array: Array of eigenvalues for each k-point in format output by STARLIGHT's observables.calc_bands_kx
    """
    if not figsize:
        figsize=(6,4)
    plt.figure(figsize=figsize)
    if not kx_range:
        kx_range = [-np.pi,np.pi]
    kx_min, kx_max = kx_range
    if not ener_range:
        scale = np.abs(eigvals_array.max()-eigvals_array.min())*0.1
        ener_range = [eigvals_array.min()-scale, eigvals_array.max()+scale]
    emin, emax = ener_range
    for band_values in eigvals_array:
        plt.plot(kx_values, band_values, color='b', linewidth=0.8)
    
    if hline:
        plt.axhline(y=hline, color='orange', linestyle='--')
    plt.xticks([-np.pi, 0, np.pi], [f'$-\\pi$', f'0', f'$\\pi$'], fontsize=12) 
    plt.xlim(kx_min,kx_max)
    plt.yticks(fontsize=12); plt.ylim(emin,emax)
    plt.xlabel(r'$k_x$', fontsize=16)
    plt.ylabel(r'Energy ($t$)', fontsize=16)
    if save_png:
        plt.savefig(save_png, bbox_inches="tight", dpi=600)
    else:
        plt.figure(dpi=600)
    if display:
        plt.show()

def plot_bands_ky(ky_values, eigvals_array, q=False, ky_range=False, ener_range=False, figsize=False, display=False, save_png=False):
    """ Function which plots band structure along ky.
        PARAMETERS
         ky_values: List of ky points at which energies were evaluated.
     eigvals_array: Array of eigenvalues for each k-point in format output by STARLIGHT's observables.calc_bands_ky
    """
    if not figsize:
        figsize=(6,4)
    plt.figure(figsize=figsize)
    if not ky_range:
        ky_range = [-np.pi,np.pi]
    ky_min, ky_max = ky_range
    if not ener_range:
        scale = np.abs(eigvals_array.max()-eigvals_array.min())*0.1
        ener_range = [eigvals_array.min()-scale, eigvals_array.max()+scale]
    emin, emax = ener_range 
    for band_values in eigvals_array:
        plt.plot(ky_values, band_values, color='b', linewidth=0.8)
        #plt.plot(ky_values, band_values, linewidth=0.8)
    if q:
        plt.xticks([-np.pi/q, 0, np.pi/q], [f'$-\\pi/{q}$', f'0', f'$\\pi/{q}$'], fontsize=12) 
    plt.xlim(ky_min,ky_max)
    plt.yticks(fontsize=12); plt.ylim(emin,emax)
    plt.xlabel(r'$k_y^0$', fontsize=16)
    plt.ylabel(r'Energy ($t$)', fontsize=16)
    if save_png:
        plt.savefig(save_png, bbox_inches="tight", dpi=600)
    else:
        plt.figure(dpi=600)
    if display:
        plt.show()
