"""
  Filename: observables.py
    Author: Daniel Staros
   Summary: Contains functions for calculating tight-binding observables from real-space and k-space models. 
            Currently supported observables are the 1RDM, occupations, site renormalization energies band structures, 
            and conductivity (xx, yy, and xy).
   Imports: math, numpy, starlight.lattice, starlight.fermi.fermi_func,
            starlight.utilities.print_time, starlight.velocity.*, starlight.gutzwiller.*
"""

import math
import numpy as np
from math import e
from starlight import lattice
from starlight.fermi import fermi_func
from starlight.utilities import print_time
from starlight.velocity import *
from starlight.gutzwiller import *
from starlight.hamiltonian import ham_plaq

def rdm1(eigvals, eigvecs, mu, T=False):
    """ Calculates the one-body reduced density matrix (1RDM)
        PARAMETERS
           eigvals: Eigenvalues obtained by diagonalizing a tight-binding Hamiltonian
           eigvecs: Eigenvectors obtained by diagonalizing a tight-binding Hamiltonian
                mu: Fermi level
                 T: Temperature in units of t (default corresponds to kBT = 0.01 t)
    """
    fermi_occupations = np.array([fermi_func(eigval, mu, T=T) for eigval in eigvals])
    fermi_diag = np.diag(fermi_occupations)
    rho = eigvecs @ fermi_diag @ np.conj(eigvecs.T)
    return rho

def calc_occups(*args, **kwargs):
    """ 
       Calculates occupations from the 1RDM
    """
    rho = rdm1(*args, **kwargs)
    occups = np.array([rho[i][i].real for i in range(len(rho))])
    return occups

def calc_derivs(ta, tb, d_params, rdm1, e_list=False, q=False, gauge=False, real_space=False):
    """
       Helper function that calculates the derivatives to be fed into the level renormalization.
    """
    dsqa_dns, symbols_dict = deriv_sq_a_sp('n_s')

    # Need to return four different lists of derivatives
    dsqa_sig_dn_sig_list  = []
    dsqa_sigb_dn_sig_list = []

    if real_space:
        site_len = len(rdm1)
        bi = int(len(rdm1)/2)
    else:
        site_len = 2*q
        bi = q

    for j in range(site_len):

        # Obtaining corresponding derivative
        d_j = d_params[j % int(len(rdm1)/2)]
        njs_index     = j
        njs_bar_index = (((j+int(len(rdm1)/2)))) % len(rdm1)
        dsqa_sig_dn_sig  = np.complex128(dsqa_dns.subs({symbols_dict['n_s']: rdm1[njs_index][njs_index],
                                                 symbols_dict['n_sbar']: rdm1[njs_bar_index][njs_bar_index],
                                                 symbols_dict['d']: d_j}).evalf())
        dsqa_sigb_dn_sig = np.complex128(dsqa_dns.subs({symbols_dict['n_s']: rdm1[njs_bar_index][njs_bar_index],
                                                 symbols_dict['n_sbar']: rdm1[njs_index][njs_index],
                                                 symbols_dict['d']: d_j}).evalf())
        dsqa_sig_dn_sig_list.append(dsqa_sig_dn_sig)
        dsqa_sigb_dn_sig_list.append(dsqa_sigb_dn_sig)

    # Define final derivative lists
    dsqa_js_dn_js   = dsqa_sig_dn_sig_list[:bi]  # First set for lambda_js
    dsqa_jsb_dn_js  = dsqa_sigb_dn_sig_list[:bi] # Second set for lambda_js
    dsqa_jsb_dn_jsb = dsqa_sig_dn_sig_list[bi:]  # First set for lambda_jsbar
    dsqa_js_dn_jsb  = dsqa_sigb_dn_sig_list[bi:] # Second set for lambda_jsbar
    
    return dsqa_js_dn_js, dsqa_jsb_dn_js, dsqa_jsb_dn_jsb, dsqa_js_dn_jsb

def calc_lambdas(Nx, Ny, ta, tb, d_params, rdm1, deriv_set, e_list=False, q=False, gauge=False, real_space=False):
    """
       Function for calculating level renormalization during self-consistent Gutzwiller optimization.
    """
    # Set default q (magnetic flux parameter) if necessary
    if q:
        if (gauge == 'L1') or (gauge == False):
            if ((math.gcd(Ny, q) != 1) and (int(Ny) >= int(q))) or (int(Ny) == int(q)) or (q == 1): # Ensure q is physically allowed
                phi = 1/q
                flux_L1 = -1j*2*np.pi*phi
                flux_L2 = 0
            else:
                raise ValueError("q must be commensurate with, but not equal to, Nx in the 1st Landau gauge (L1).")
    else:
        phi = 0

    if q == 1:
        print("CAUTION: Setting q to 1 is only compatible in k-space optimization with p/q = 0/1 = 0.")

    # Splitting RDM and derivative list (bi means "bisect" index)
    bi = int(len(rdm1)/2)
    rdm1_s    = rdm1[:bi,:bi] # RDM for s block
    rdm1_sbar = rdm1[bi:,bi:] # RDM for sbar block

    if real_space:
        dsqa_js_dn_js = deriv_set[0]   # Length is Nx*Ny
        dsqa_jsb_dn_js = deriv_set[1]  # Length is Nx*Ny
        dsqa_jsb_dn_jsb = deriv_set[2] # Length is Nx*Ny
        dsqa_js_dn_jsb = deriv_set[3]  # Length is Nx*Ny
    else:
        dsqa_js_dn_js = deriv_set[0][:q]
        dsqa_jsb_dn_js = deriv_set[1][:q]
        dsqa_jsb_dn_jsb = deriv_set[2][:q]
        dsqa_js_dn_jsb = deriv_set[3][:q]

    lambdas_s  = []
    lambdas_sb = []
    sites = lattice.site_list(Nx,Ny)
    if real_space:
        site_loop = sites
    else:
        site_loop = sites[:q]

    for m,n in site_loop:
        # Getting indices for neighbor hops
        j = (n % Ny)+(Ny*(m % Nx))
        j_mp1 = (n % Ny)+(Ny*((m+1) % Nx))
        j_np1 = ((n+1) % Ny)+(Ny*(m % Nx))

        # Getting hopping expectations
        m_hop_up_s  = rdm1_s[j][j_mp1]; m_hop_up_sb = rdm1_sbar[j][j_mp1]
        m_hop_dw_s  = rdm1_s[j_mp1][j]; m_hop_dw_sb = rdm1_sbar[j_mp1][j]
        n_hop_up_s  = rdm1_s[j][j_np1]; n_hop_up_sb = rdm1_sbar[j][j_np1]
        n_hop_dw_s  = rdm1_s[j_np1][j]; n_hop_dw_sb = rdm1_sbar[j_np1][j]

        # Getting parameters from which sq_a's will be calculated
        d_mp1n_s    = d_params[j_mp1]
        d_mnp1_s    = d_params[j_np1]
        occ_mp1n_s  = rdm1_s[j_mp1][j_mp1]; occ_mp1n_sb = rdm1_sbar[j_mp1][j_mp1]
        occ_mnp1_s  = rdm1_s[j_np1][j_np1]; occ_mnp1_sb = rdm1_sbar[j_np1][j_np1]
        sq_a_mp1n_s = sq_a(occ_mp1n_s, occ_mp1n_sb, d_mp1n_s); sq_a_mp1n_sb  = sq_a(occ_mp1n_sb, occ_mp1n_s, d_mp1n_s)
        sq_a_mnp1_s = sq_a(occ_mnp1_s, occ_mnp1_sb, d_mnp1_s); sq_a_mnp1_sb  = sq_a(occ_mnp1_sb, occ_mnp1_s, d_mnp1_s)

        # Calculating lambda_s terms
        theta_n     = 2*np.pi*n*phi
        deriv_term_ss   = dsqa_js_dn_js[j]*((ta * e**(-1j*theta_n) * sq_a_mp1n_s * m_hop_up_s)+(tb * sq_a_mnp1_s * n_hop_up_s))
        deriv_term_sbs  = dsqa_jsb_dn_js[j]*((ta * e**(-1j*theta_n) * sq_a_mp1n_sb * m_hop_up_sb)+(tb * sq_a_mnp1_sb * n_hop_up_sb))
        deriv_term_sbsb = dsqa_jsb_dn_jsb[j]*((ta * e**(-1j*theta_n) * sq_a_mp1n_sb * m_hop_up_sb)+(tb * sq_a_mnp1_sb * n_hop_up_sb))
        deriv_term_ssb  = dsqa_js_dn_jsb[j]*((ta * e**(-1j*theta_n) * sq_a_mp1n_s * m_hop_up_s)+(tb * sq_a_mnp1_s * n_hop_up_s))
        deriv_sum_lams  = deriv_term_ss + deriv_term_sbs
        deriv_sum_lamsb = deriv_term_sbsb + deriv_term_ssb

        if not e_list:
            lambda_mn_s  = deriv_sum_lams + np.conj(deriv_sum_lams); lambdas_s.append(lambda_mn_s)
            lambda_mn_sb = deriv_sum_lamsb + np.conj(deriv_sum_lamsb); lambdas_sb.append(lambda_mn_sb)
        else:
            lambda_mn_s  = -e_list[j] + deriv_sum_lams + np.conj(deriv_sum_lams); lambdas_s.append(lambda_mn_s)
            lambda_mn_sb = -e_list[j] + deriv_sum_lamsb + np.conj(deriv_sum_lamsb); lambdas_sb.append(lambda_mn_sb)

    lambdas = lambdas_s+lambdas_sb
    return lambdas

def calc_bands_kx(ta, tb, kx_min, kx_max, kx_int, ky, p, q, occupations=False, d_params=False, e_list=False):
    """ Calculates band structure along kx
        PARAMETERS
                ta: Quantifies hopping energy in x
                tb: Quantifies hopping energy in y
            kx_min: Minimum value of kx range
            kx_max: Maximum value of kx range
            kx_int: Size of dkx intervals
                ky: ky point at which to evaluate kx disperion
                 p: An integer defining magnetic flux (must be relatively prime with q)
                 q: An integer defining magnetic flux (must be relatively prime with p)
       occupations: List of site occupations defining Gutzwiller parameters (length = q)
          d_params: Array of double occupancy parameters defining Gutzwiller parameters (length = q)
            e_list: List of site energies
    """
    kx_values = list(np.arange(kx_min,kx_max,kx_int))
    eigvals_list = []
    counter = 0
    for kx in kx_values:
        counter += 1
        ham_kx_s, ham_kx_sb = ham_plaq(ta=ta, tb=tb, kx=kx, ky=ky, p=p, q=q, occupations=occupations, d_params=d_params, e_list=e_list)
        eigvals_kx_s, eigvecs_kx_s = np.linalg.eigh(ham_kx_s)
        eigvals_kx_sb, eigvecs_kx_sb = np.linalg.eigh(ham_kx_sb)
        eigvals_list.append(list(eigvals_kx_s)+list(eigvals_kx_sb))
        eigvals_array = np.array(eigvals_list).T

    return kx_values, eigvals_array

def calc_bands_ky(ta, tb, kx, ky_min, ky_max, ky_int, p, q, occupations=False, d_params=False, e_list=False):
    """ Calculates band structure along ky
        PARAMETERS
                ta: Quantifies hopping energy in x
                tb: Quantifies hopping energy in y
                kx: kx point at which to evaluate ky disperion
            ky_min: Minimum value of ky range
            ky_max: Maximum value of ky range
            ky_int: Size of dky intervals
                 p: An integer defining magnetic flux (must be relatively prime with q)
                 q: An integer defining magnetic flux (must be relatively prime with p)
       occupations: List of site occupations defining Gutzwiller parameters (length = q)
          d_params: Array of double occupancy parameters defining Gutzwiller parameters (length = q)
            e_list: List of site energies
    """
    ky_values = list(np.arange(ky_min,ky_max,ky_int))
    eigvals_list = []
    counter = 0
    for ky in ky_values:
        counter += 1
        ham_ky_s, ham_ky_sb = ham_plaq(ta=ta, tb=tb, kx=kx, ky=ky, p=p, q=q, occupations=occupations, d_params=d_params, e_list=e_list)
        eigvals_ky_s, eigvecs_ky_s = np.linalg.eigh(ham_ky_s)
        eigvals_ky_sb, eigvecs_ky_sb = np.linalg.eigh(ham_ky_sb)
        eigvals_list.append(list(eigvals_ky_s)+list(eigvals_ky_sb))
        eigvals_array = np.array(eigvals_list).T

    return ky_values, eigvals_array

def sigma_xy_range(Nx, Ny, ta, tb, mu_min, mu_max, eigvals, eigvecs, e_list=False, q=False, gauge=False, occupations=False, d_params=False, mu_int=False, d=False, T=False, print_data=False):
    """ Function which calculates the transverse conductivity of a real-space model.
        PARAMETERS
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            mu_min: Lower bound on mu for conductivity evaluation
            mu_max: Upper bound on mu for conductivity evaluation
           eigvals: Eigenvalues obtained by diagonalizing a real-space model Hamiltonian
           eigvecs: Eigenvectors obtained by diagonalizing a real-space model Hamiltonian
          e_list: List of site energies per site/row
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]))
       occupations: List of average occupations <c_i^{dagger} c_i> for each site i (length = Ny)
          d_params: Array of double occupancy parameters d_i for each site i        (length = Ny)
            mu_int: Chemical potential grid granularity
                 d: Level broadening in energy units defined by the model
                 T: Temperature in unites of Kelvin
        print_data: Option to print kx evaluation updates, kx values and band energies 
    """
    print_time('start','conductivity')
    #Nstates = Nx*Ny*2
    Nstates = Nx*Ny
    ec = 1; h = 1; hbar = h/(2*np.pi)
    if not d:                                              # Set default d (level broadening) if necessary
        d = 0.1
    if not mu_int:                                         # Set default mu_int (mesh granularity) for QHC evaluation
        mu_int = (mu_max-mu_min)/800

    print("\n##### BEGINNING TRANSVERSE CONDUCTIVITY CALCULATION\n")
    # Define velocity operators; only calculate for spin-up block since spin blocks are identical (for now).
    vx_mat = vx(Nx, Ny, ta, tb, e_list=e_list, q=q, gauge=gauge, occupations=occupations, d_params=d_params)[:Nx*Ny,:Nx*Ny]
    vy_mat = vy(Nx, Ny, ta, tb, e_list=e_list, q=q, gauge=gauge, occupations=occupations, d_params=d_params)[:Nx*Ny,:Nx*Ny]

    # Initiate calculation of transverse quantum Hall conductivity
    sum_ab = 0+0j
    eigdat = list(zip(eigvals, eigvecs))
    outer_list = []
    print("Calculating sigma_xy over the range of mu for each nonequal pair of eigenvectors...")
    for eigval_a, eigvec_a in eigdat:
        for eigval_b, eigvec_b in eigdat:
            if np.array_equal(eigvec_a, eigvec_b):
                pass
            else:
                E_a = eigval_a; E_b = eigval_b
                ener_diff  = E_a-E_b

                # Define velocity operator expectation values
                vx_ab  = np.matmul(np.conj(eigvec_a).T,np.matmul(vx_mat,eigvec_b))
                vy_ba  = np.matmul(np.conj(eigvec_b).T,np.matmul(vy_mat,eigvec_a))

                # Defining the summand for given set α,β
                sigma_mus = []
                for m in range(math.floor(mu_min/mu_int), math.floor(mu_max/mu_int), 1): # Generate list corresponding to given summand evaluated over a range of mu
                    mu = m*mu_int
                    fermi_diff  = fermi_func(E_a,mu)-fermi_func(E_b,mu)
                    numerator   = vx_ab*vy_ba
                    denominator = ((ener_diff)**2)+(d**2)
                    fraction    = numerator/denominator
                    summand     = fermi_diff*fraction

                    sigma_mus.append(summand)
                outer_list.append(sigma_mus)

    outer_array = np.array(outer_list)
    term_array = np.sum(outer_array, axis=0) # Sum over summands for each given mu; generates list of QHC values for every mu

    norm = 1j*(ec**2)*hbar/Nstates
    sigma_array = norm*term_array

    mu_list = []
    for m in range(math.floor(mu_min/mu_int), math.floor(mu_max/mu_int), 1): # Generate list corresponding to given summand evaluated over a range of mu
        mu = m*mu_int
        mu_list.append(mu)

    print("\nSUCCESS: Conductivity calculation complete.\n\n")
    print("##### DATA SUMMARY")
    print("mu_min, mu_max: [", min(mu_list), ", ", max(mu_list),"]")
    print("sigma_xy_min, sigma_xy_max: [", np.round(np.real(sigma_array).min(),3), ", ",  np.round(np.real(sigma_array).max(),3),"]")
    imag_sum = np.sum(np.imag(sigma_array))
    print("sigma_xy total sum of ( real, imaginary ) components: (",np.sum(np.real(sigma_array)),", ",imag_sum,")\n")

    if not print_data:
        return mu_list, list(np.real(sigma_array))
    else:
        print("##### RAW DATA\n")
        print("Printing real components of sigma_xy_array... \n",repr(list(np.real(sigma_array))),"\n")
        if imag_sum > 0.1:
            print("WARNING: Imaginary component of sigma_xy is nonzero.\n")
            print("Printing imaginary components of sigma_xy_array... \n",np.imag(sigma_array),"\n")
        else:
            print("Imaginary components of sigma_xy are negligible, so the corresponding array is not being printed... \n")
        print("Printing mu_list... \n", repr(list(np.round(mu_list,5))), "\n")
    print_time('end','conductivity')

# WARNING SPIN NOT YET ADDED - 5/6/25
def sigma_longitud_range(Nx, Ny, ta, tb, mu_min, mu_max, eigvals, eigvecs, direction=False, e_list=False, q=False, gauge=False, occupations=False, d_params=False, mu_int=False, d=False, T=False, print_data=False):
    """ Function which calculates the longitudinal conductivity of a real-space model.
        PARAMETERS
                Nx: Effective number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            mu_min: Lower bound on mu for conductivity evaluation
            mu_max: Upper bound on mu for conductivity evaluation
           eigvals: Eigenvalues obtained by diagonalizing a real-space model Hamiltonian
           eigvecs: Eigenvectors obtained by diagonalizing a real-space model Hamiltonian
         direction: Direction along which to evaluate longitudinal conductivity (options are 'x' or 'y')
          e_list: List of site energies per site/row
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0]))
       occupations: List of average occupations <c_i^{dagger} c_i> for each site i (length = Ny)
          d_params: Array of double occupancy parameters d_i for each site i        (length = Ny)
            mu_int: Chemical potential grid granularity
                 d: Level broadening in energy units defined by the model
                 T: Temperature in unites of Kelvin
        print_data: Option to print kx evaluation updates, kx values and band energies 
    """
    print_time('start','conductivity')
    Ntot = Nx*Ny
    ec = 1; h = 1; hbar = h/(2*np.pi)
    if not d:                                              # Set default d (level broadening) if necessary
        d = 0.1
    if not mu_int:                                         # Set default mu_int (mesh granularity) for QHC evaluation
        mu_int = (mu_max-mu_min)/800
    if not direction:
        direction = 'y'

    print("\n##### BEGINNING LONGITUDINAL CONDUCTIVITY CALCULATION\n")
    # Define velocity operator; only calculate for spin-up block since spin blocks are identical (for now).
    if direction == 'x':
        v_mat = vx(Nx, Ny, ta, tb, e_list=e_list, q=q, gauge=gauge, occupations=occupations, d_params=d_params)[:Nx*Ny,:Nx*Ny]
    elif direction == 'y':
        v_mat = vy(Nx, Ny, ta, tb, e_list=e_list, q=q, gauge=gauge, occupations=occupations, d_params=d_params)[:Nx*Ny,:Nx*Ny]

    # Initiate calculation of transverse quantum Hall conductivity
    sum_ab = 0+0j
    eigdat = list(zip(eigvals, eigvecs))
    outer_list = []
    print(f"Calculating sigma_{direction*2} over the range of mu for each nonequal pair of eigenvectors...")
    for eigval_a, eigvec_a in eigdat:
        for eigval_b, eigvec_b in eigdat:
            if np.array_equal(eigvec_a, eigvec_b):
                pass
            else:
                E_a = eigval_a; E_b = eigval_b
                ener_diff = E_a-E_b
                if ener_diff != 0:

                    # Define velocity operator expectation values
                    v_ab  = np.matmul(np.conj(eigvec_a).T,np.matmul(v_mat,eigvec_b))
                    v_ba  = np.matmul(np.conj(eigvec_b).T,np.matmul(v_mat,eigvec_a))

                    # Defining the summand for given set α,β
                    sigma_mus = []
                    for m in range(math.floor(mu_min/mu_int), math.floor(mu_max/mu_int), 1): # Generate list of given summand evaluated over a range of mu
                        mu = m*mu_int
                        fermi_diff = fermi_func(E_a,mu)-fermi_func(E_b,mu)
                        numerator1 = fermi_diff; denominator1 = ener_diff
                        numerator2 = d*(v_ab*v_ba); denominator2 = ((ener_diff)**2)+(d**2)
                        fraction1  = (numerator1)/(denominator1)
                        fraction2  = (numerator2)/(denominator2)
                        summand    = (fraction1)*(fraction2)

                        sigma_mus.append(summand)
                    outer_list.append(sigma_mus)

                else:
                    sigma_mus = []
                    for m in range(math.floor(mu_min/mu_int), math.floor(mu_max/mu_int), 1):
                        summand = 0
                        sigma_mus.append(summand)
                    outer_list.append(sigma_mus)

    outer_array = np.array(outer_list)
    term_array = np.sum(outer_array, axis=0) # Sum over summands for each given mu; generates list of QHC values for every mu

    norm = -(ec**2)*hbar/Ntot
    sigma_array = norm*term_array

    mu_list = []
    for m in range(math.floor(mu_min/mu_int), math.floor(mu_max/mu_int), 1): # Generate list corresponding to given summand evaluated over a range of mu
        mu = m*mu_int
        mu_list.append(mu)

    print("\nSUCCESS: Conductivity calculation complete.\n\n")
    print("##### DATA SUMMARY")
    print("mu_min, mu_max: [", min(mu_list), ", ", max(mu_list),"]")
    print(f"sigma_{direction*2}_min, sigma_{direction*2}_max: [", np.round(np.real(sigma_array).min(),3), ", ",  np.round(np.real(sigma_array).max(),3),"]")
    imag_sum = np.sum(np.imag(sigma_array))
    print(f"sigma_{direction*2} total sum of ( real, imaginary ) components: (",np.sum(np.real(sigma_array)),", ",imag_sum,")\n")
    if not print_data:
        return mu_list, list(np.real(sigma_array))
    else:
        print("##### RAW DATA\n")
        print(f"Printing real components of sigma_{direction*2}_array... \n",repr(list(np.real(sigma_array))),"\n")
        if imag_sum > 0.1:
            print(f"WARNING: Imaginary component of sigma_{direction*2} is nonzero.\n")
            print(f"Printing imaginary components of sigma_{direction*2}_array... \n",np.imag(sigma_array),"\n")
        else:
            print("Imaginary components of sigma_xy are negligible, so the corresponding array is not being printed... \n")
        print("Printing mu_list... \n", repr(list(np.round(mu_list,5))), "\n")
    print_time('end','conductivity')
