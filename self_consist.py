"""
  Filename: self_consist.py
    Author: Daniel Staros
   Summary: Contains functions for self-consistent solutions of Gutzwiller-renormalized lattice models.
   Imports: time, numpy, math.e, starlight.*, scipy.optimize.*, datetime.datetime
"""

import time
import numpy as np
from math import e
from starlight import *
from scipy.optimize import *
from datetime import datetime
from starlight.fermi import *
from starlight.hamiltonian import real_ham, ham_plaq, expectation_muVT
from starlight.utilities import print_time

def callback(xk, *args):
    ''' Function for checking progress at each scipy subiteration '''
    print(f"\n[d_list] at current subiteration: {xk}, time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

def expectation_obj(vars, Nx, Ny, ta, tb, e_list=False, U=False, q=False, gauge=False, occupations=False, filling=False, T=False):
    ''' Expectation value objective function for real-space self-consistent optimization.
        PARAMETERS
              vars: Variables to be optimized by scipy (double occupancy parameters d_j); length = Nx*Ny
                Nx: Number of lattice sites in x-direction
                Ny: Number of lattice sites in y-direction
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
            e_list: List of onsite energies
                 U: Effective onsite Coulomb repulsion (units of t)
                 q: Integer defining the magnitude of magnetic flux
             gauge: Landau gauge (options: 'L1' (A=[-y*B_z,0,0]) or 'L2' (A=[0,x*B_z,0])); default is L1
       occupations: List of average occupations <c_{jσ}^{dagger} c_{jσ}> per spin site jσ (length = Nx * Ny * 2)
    '''
    hamiltonian      = real_ham(Nx, Ny, ta, tb, e_list=e_list, q=q, gauge=gauge, occupations=occupations, d_params=vars)
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    mu               = mu_calc(Nx, Ny, eigvals, filling, T=T)
    expect_kin       = expectation_muVT(eigvals, mu, T=T)
    if len(vars) == 1:
        hubbard_ener = U*np.sum(vars)*Nx*Ny
    else:
        hubbard_ener = U*np.sum(vars)
    expectation_tot  = np.sum([expect_kin,hubbard_ener]).real
    return expectation_tot

def expectation_plaq_ham(vars, Nx, Ny, ta, tb, p, q, U, occupations=False, e_list=False, filling=False, T=False):
    ''' Expectation value objective function for "mixed space" self-consistent optimization in L1 gauge.
        PARAMETERS
              vars: Variables to be optimized by scipy (double occupancy parameters d_j); length = Nx*Ny
                Nx: Number of pseudo-lattice sites in x-direction (sets kx mesh)
                Ny: Number of unit cells in y-direction (sets ky) - defined as: (number of pseudo-lattice sites)/(number of plaquettes per cell, q)
                ta: Quantifies hopping energy along the x-direction
                tb: Quantifies hopping energy along the y-direction
                 p: An integer defining magnetic flux (must be relatively prime with q); default is 1
                 q: An integer defining magnetic flux (must be relatively prime with p)
            e_list: List of onsite energies
                 U: Effective onsite Coulomb repulsion (units of t)
                 q: Integer defining the magnitude of magnetic flux
       occupations: List of average occupations <c_{jσ}^{dagger} c_{jσ}> per spin site jσ (length = Nx * Ny * 2)
    '''
    # Define k-grid corresponding to real-space lattice
    kx_points = np.arange(-Nx/2, Nx/2, 1)*(np.pi/(Nx/2))
    ky_points = np.arange(-Ny/(2*q), Ny/(2*q), 1)*(np.pi/(Ny/(2*q)))

    # Evaluate eigenvalues of Hamiltonian
    eigvals = []
    for kx in kx_points:
        for ky in ky_points:
            ham_ks, ham_ksb = ham_plaq(ta, tb, kx, ky0=ky, p=p, q=q, occupations=occupations, d_params=vars, e_list=e_list)
            eigvals_ks, eigvecs_ks   = np.linalg.eigh(ham_ks)
            eigvals_ksb, eigvecs_ksb = np.linalg.eigh(ham_ksb)
            eigvals += list(eigvals_ks)+list(eigvals_ksb)
    eigvals.sort()

    # Calculate energy
    mu       = mu_calc(Nx, Ny, eigvals, filling, T=T)
    ekin     = expectation_muVT(eigvals, mu) # Calculate kinetic energy
    epot     = U*len(kx_points)*len(ky_points)*np.sum(vars) # Calculate potential energy
    etot     = ekin + epot # Calculate total energy
    return etot

def sc_min(obj_func, args, param_guess=False, bounds=False, rdm0=False, min_method=False, gtoler=False, d_tol=False, max_cycles=False, iteration=False, prev_obj=False, mixing=False, verbose=False, glob_min=False, isotropic=False):
    # Objective function options
    if obj_func == expectation_obj: # Real-space
        Nx, Ny, ta, tb, e_list, U, q, gauge, occupations, filling, T = args
        Ntot = Nx*Ny
        if q:
            if q == 1:
                raise ValueError("q and Ny must be relatively prime.")
        if (q is False) and (q is not 0):
            pass
        elif (q == 0) and (q is not False):
            raise ValueError("q must NOT be 0. This corresponds to divergent magnetic flux.")
    elif obj_func == expectation_plaq_ham: # k-space ("mixed space"); this one doesn't have a gauge choice option yet.
        Nx, Ny, ta, tb, p, q, U, occupations, e_list, filling, T = args
        Ntot = Nx*Ny

    if not iteration:
        iteration = 0
        print_time('start','gutzwiller')
        print("\n##### BEGINNING SELF CONSISTENCY ITERATION No. "+str(iteration))
        iteration += 1
    else:
        print("\n##### BEGINNING SELF CONSISTENCY ITERATION No. "+str(iteration))
        iteration += 1

    if not d_tol:
        d_tol = 0.005*0.25 # Setting to a reasonable value for now. 0.001 is a small change in d.
    if not max_cycles:
        max_cycles = 10
    if not min_method:
        min_method = 'trust-constr'
    if not filling:
        filling = 0.5
        print("Filling not specified. Assuming half filling (filling = 0.5)...")
    if not gtoler:
        if isotropic:
            gtoler = 1e-8
        else:
            gtoler = 1e-6
    if mixing:
        print("Mixing set by user: ",mixing)

    # Calculate bare occupations
    if not occupations:
        print("Initial occupations not provided. Initializing based on uncorrelated (bare) Hamiltonian...")
        # Executed when preparing to optimize Gutzwiller parameters fully in real-space
        if obj_func == expectation_obj:
            ham_init = real_ham(Nx, Ny, ta, tb, e_list, q)
            eigvals, eigvecs = np.linalg.eigh(ham_init)
            mu = mu_calc(Nx, Ny, eigvals, filling, T=T)
            rdm_r0  = rdm1(eigvals, eigvecs, mu)
            occupations = list(calc_occups(eigvals, eigvecs, mu))
            ns_list = occupations[:Nx*Ny]; nsbar_list = occupations[Nx*Ny:]
            ni_list = [ns_list[i]+nsbar_list[i] for i in range(len(ns_list))]
            print("Occupation initialization successful. Bare occupations are: ns = ",np.round(ns_list,10),", nsbar = ",np.round(nsbar_list,10), ", ni = ",np.round(ni_list,10))
            if isotropic:
                print("Utilizing isotropic Gutzwiller approximation...")
                occupations = [occupations[0]]
                ni_list     = [ni_list[0]]
            args = Nx, Ny, ta, tb, e_list, U, q, gauge, occupations, filling, T
            if not bounds:
               if isotropic:
                    bounds = [[0.001,(ni_list[i]**2)/4] for i in range(len(ni_list))]
               else:
                    bounds = [[0.0,(ni_list[i]**2)/4] for i in range(len(ni_list))]
               print("Double occupancy bounds initialization successful. Bounds are: ",bounds)
            if not param_guess:
                param_guess = np.array([bounds[i][1] for i in range(len(bounds))])

        # Executed when preparing to optimize Gutzwiller parameters in k-space
        elif obj_func == expectation_plaq_ham:
            if p == 0:
                ham_init = real_ham(Nx, Ny, ta, tb, q=0, e_list=e_list)
            else:
                ham_init = real_ham(Nx, Ny, ta, tb, q=q, e_list=e_list) # Hacky again...
            eigvals, eigvecs = np.linalg.eigh(ham_init)
            mu = mu_calc(Nx, Ny, eigvals, filling, T=T)
            rdm_r0  = rdm1(eigvals, eigvecs, mu)
            occs_r0 = calc_occups(eigvals, eigvecs, mu)
            bare_occs_s  = list(occs_r0[0:q])
            bare_occs_sb = list(occs_r0[Ntot:Ntot+q])
            bare_occs_i  = [bare_occs_s[i]+bare_occs_sb[i] for i in range(q)]
            bare_occs    = bare_occs_s + bare_occs_sb
            print("Occupation initialization successful. Bare occupations are: ns = ",np.round(bare_occs_s,10),", nsbar = ",np.round(bare_occs_s,10), ", ni = ",np.round(bare_occs_i,10))
            args = Nx, Ny, ta, tb, p, q, U, bare_occs, e_list, filling, T
            if not bounds:
                bounds = [[0.0,(bare_occs_i[i]**2)/4] for i in range(len(bare_occs_i))]
                print("Double occupancy bounds initialization successful. Bounds are: ",*np.round(bounds,10))
            if not param_guess:
                param_guess = np.array([bounds[i][1] for i in range(len(bounds))])

    # Execute scipy minimize to find optimal values of d_{j} starting from current guess
    if min_method in ['trust-constr','slsqp']:
        if (not glob_min) and verbose:
            result = minimize(obj_func, param_guess, bounds=bounds, args=args, method=min_method, callback=callback, options = {'gtol': gtoler, 'xtol': 1e-4})
        elif (not glob_min) and (not verbose):
            result = minimize(obj_func, param_guess, bounds=bounds, args=args, method=min_method, callback=None, options = {'gtol': gtoler, 'xtol': 1e-4})
        else:
            result = shgo(obj_func, bounds=bounds, args=args, callback=callback)
    else:
        print("ERROR: The requested optimization method does not support particle number constraints. Exiting...")
        print_time('end','gutzwiller')
        sys.exit()

    # Check for solution once minimize terminates
    if result.success:
        print("\nSolution found.\nScipy output:")
        print(result,"\n")

        # Print average values of parameters
        d_list    = result.x.tolist()
        if not isotropic:
            print("Average value of double occupancies d_{j}: ",np.average(d_list))
        new_obj   = result.fun

        # Exit after maximum number of self-consistency cycles
        if (iteration == max_cycles):
            print("Maximum number of self-consistency cycles reached. Exiting...")
            print_time('end','gutzwiller')
            sys.exit()

        # Calculate the difference between the previous and current objective function values to determine whether self-consistency is achieved
        if prev_obj:
            new_ds = result.x
            delta_ds = [np.abs(new_ds[i]-prev_obj[i]) for i in range(len(new_ds))]
            shift_ds = np.roll(new_ds,1)
            delta_sh = [np.abs(shift_ds[i]-prev_obj[i]) for i in range(len(shift_ds))]
            #print("DELTAS: ", delta_sh)

            if all(delta_d < d_tol for delta_d in delta_ds) or all(delta_d < d_tol for delta_d in delta_sh):
                print("\n\n--------------------------------------------------------------------------------------------------------------------------------------------")
                print("\n\nSUCCESS: Self-consistent optimization complete.\n\n")
                if obj_func == expectation_obj:
                    ham_upd = real_ham(Nx, Ny, ta, tb, e_list, q, occupations=occupations, d_params=new_ds)
                    eigvals, eigvecs = np.linalg.eigh(ham_upd)
                    mu = mu_calc(Nx, Ny, eigvals, filling, T=T)
                    derivs_set = calc_derivs(ta, tb, new_ds, rdm0, q=q, real_space=True)
                    final_lambdas = calc_lambdas(Nx, Ny, ta, tb, new_ds, rdm0, derivs_set, q=q, real_space=True)
                    occupations = [rdm0[i][i] for i in range(len(rdm0))]
                    ham_upd = real_ham(Nx, Ny, ta, tb, q=q, e_list=final_lambdas, occupations=occupations, d_params=new_ds)
                    eigvals, eigvecs = np.linalg.eigh(ham_upd)
                    mu = mu_calc(Nx, Ny, eigvals, filling=filling, T=T)
                    final_occupations = calc_occups(eigvals, eigvecs, mu, T=T)
                elif obj_func == expectation_plaq_ham:
                    # Getting final occupations of renormalized Hamiltonian
                    d_tile  = np.array(list(new_ds)*int(Ny/q)*Nx)
                    derivs_set = calc_derivs(ta, tb, d_tile, rdm0, q=q)
                    final_lambdas = calc_lambdas(Nx, Ny, ta, tb, d_tile, rdm0, derivs_set, q=q)
                    lam_tile = list(np.array(final_lambdas*int(Ny/q)*Nx))
                    occupations = [rdm0[i][i] for i in range(len(rdm0))]
                    if p == 0:
                        ham_upd = real_ham(Nx, Ny, ta, tb, q=0, e_list=lam_tile, occupations=occupations, d_params=d_tile)
                    else:
                        ham_upd = real_ham(Nx, Ny, ta, tb, q=q, e_list=lam_tile, occupations=occupations, d_params=d_tile) # Hacky again...
                    eigvals, eigvecs = np.linalg.eigh(ham_upd)
                    mu = mu_calc(Nx, Ny, eigvals, filling=filling, T=T)
                    final_occupations = calc_occups(eigvals, eigvecs, mu, T=T)
                    final_lambdas = final_lambdas[:2*q]
                print("FERMI LEVEL : ",mu," (units of t)")
                print("TOTAL ENERGY: ",new_obj," (units of t)\n\n")
                print("Total number of self-consistent iterations: ",iteration)
                print("Absolute difference of last two d parameter lists: ", delta_ds, "\n")
                print("Final scipy gradient (grad): ",result.grad, "\n")
                print("Final scipy Lagrangian gradient (lagrangian_grad): ",result.lagrangian_grad, "\n")
                print("Site renormalization energies λ_{js}: ", list(np.round(np.real(final_lambdas),5)),"\n")
                if obj_func == expectation_obj:
                    ns_list  = final_occupations[:Nx*Ny]; nsbar_list = final_occupations[Nx*Ny:]
                    nj_list  = [ns_list[j]+nsbar_list[j] for j in range(len(ns_list))]
                    bare_ns  = occupations[:Nx*Ny]
                    bare_nsb = occupations[Nx*Ny:]
                elif obj_func == expectation_plaq_ham:
                    ns_list    = list(np.real(final_occupations[0:q]))
                    nsbar_list = list(np.real(final_occupations[Ntot:Ntot+q]))
                    nj_list    = [ns_list[j]+nsbar_list[j] for j in range(q)]
                    bare_ns    = occupations[0:q]
                    bare_nsb   = occupations[Ntot:Ntot+q]
                print("Quasiparticle occupations:")
                print("n_{j}    : ", list(np.round(nj_list,5)))
                print("n_{js}   : ", list(np.round(ns_list,5)))
                print("n_{jsbar}: ", list(np.round(nsbar_list,5)),"\n")
                final_d_list = [round(d, 5) for d in list(new_ds)]
                print("Optimized double occupancies d_{j}: ", final_d_list)
                print("d_min, d_max: ",min(final_d_list),", ",max(final_d_list))
                print("d_avg: ",np.mean(final_d_list),"\n")
                final_alpha_list = [round(sq_a(bare_ns[j], bare_nsb[j], list(new_ds)[j]).real**2,5) for j in range(len(new_ds))]
                print("Optimized Gutzwiller renormalization parameters alpha_{j}: ", final_alpha_list)
                print("alpha_min, alpha_max: ",min(final_alpha_list),", ",max(final_alpha_list))
                print("alpha_avg: ",round(np.mean(final_alpha_list),5),"\n")
                print_time('end','gutzwiller')

            else:
                new_ds = result.x
                if mixing:
                    mix_params = (new_ds*(mixing)) + (param_guess*(1-mixing))
                    print("MIX PARAMS: ",mix_params)
                    new_ds = mix_params

                # Define input for and initiate next self-consistency cycle
                if obj_func == expectation_obj:
                    derivs_list = calc_derivs(ta, tb, new_ds, rdm0, q=q, real_space=True)
                    lambdas = calc_lambdas(Nx, Ny, ta, tb, new_ds, rdm0, derivs_list, q=q, real_space=True)
                    #if e_list:
                    #    energies = [lambdas[i]+e_list[i] for i in range(len(lambdas))]
                    #    args = Nx, Ny, ta, tb, energies, U, q, gauge, occupations, filling, T
                    #else:
                    #    args = Nx, Ny, ta, tb, lambdas, U, q, gauge, occupations, filling, T
                    args = Nx, Ny, ta, tb, lambdas, U, q, gauge, occupations, filling, T
                elif obj_func == expectation_plaq_ham:
                    d_tile  = np.array(list(new_ds)*int(Ny/q)*Nx) # Need to tile because I need to keep track of entire real-space RDM0
                    derivs_set = calc_derivs(ta, tb, d_tile, rdm0, q=q)
                    lambdas = calc_lambdas(Nx, Ny, ta, tb, d_tile, rdm0, derivs_set, q=q)
                    bare_occs_s  = [rdm0[i][i] for i in range(0,q)]
                    bare_occs_sb = [rdm0[i][i] for i in range(Ntot,Ntot+q)]
                    bare_occs = bare_occs_s + bare_occs_sb
                    args = Nx, Ny, ta, tb, p, q, U, bare_occs, lambdas, filling, T

                sc_min(obj_func, args, param_guess=new_ds, bounds=bounds, rdm0=rdm0, min_method=min_method, gtoler=gtoler, d_tol=d_tol, max_cycles=max_cycles, iteration=iteration, prev_obj=list(new_ds), mixing=mixing, glob_min=glob_min)

        # Initiate iteration 2 when no previous objective function exists for comparison
        else:
            new_ds = result.x

            # Define input for and initiate next self-consistency cycle
            if obj_func == expectation_obj:
                if q is False:
                    print("Final scipy gradient (grad): ",result.grad)
                    print("Final scipy Lagrangian gradient (lagrangian_grad): ",result.lagrangian_grad,"\n\n")
                    final_ham = real_ham(Nx, Ny, ta, tb, e_list, q, occupations=occupations, d_params=new_ds)
                    eigvals, eigvecs = np.linalg.eigh(final_ham)
                    mu = mu_calc(Nx, Ny, eigvals, filling, T=T)
                    print("FERMI LEVEL : ",mu," (units of t)")
                    print("TOTAL ENERGY: ",result.fun," (units of t)\n\n")
                    if len(new_ds) == 1:
                        print("Optimized double occupancy parameter d: ", new_ds[0])
                        final_alpha = round(sq_a(np.round(occupations[0],3), np.round(occupations[0],3), new_ds[0]).real**2,5)
                        print("Optimized Gutzwiller renormalization parameter alpha: ",final_alpha,"\n\n")
                    else:
                        print("Optimized double occupancies d_{j}: ", [new_ds[j] for j in range(len(new_ds))])
                        print("d_min, d_max: ",min(new_ds),", ",max(new_ds))
                        print("d_avg: ",np.mean(new_ds),"\n")
                        final_alphas = [round(sq_a(np.round(occupations[j],3), np.round(occupations[j],3), new_ds[j]).real**2,5) for j in range(len(new_ds))]
                        print("Optimized Gutzwiller renormalization parameters alpha_{j}: ", final_alphas)
                        print("alpha_min, alpha_max: ",min(final_alphas),", ",max(final_alphas))
                        print("alpha_avg: ",round(np.mean(final_alphas),5),"\n\n")
                    print_time('end','gutzwiller')

                else:
                    if len(new_ds) == 1:
                        print("Optimized double occupancy parameter d: ", new_ds[0])
                        final_alpha = round(sq_a(np.round(occupations[0],3), np.round(occupations[0],3), new_ds[0]).real**2,5)          
                        print("Optimized Gutzwiller renormalization parameter alpha: ",final_alpha,"\n\n")
                        print_time('end','gutzwiller')
                        sys.exit()
                    else:
                        derivs_set = calc_derivs(ta, tb, new_ds, rdm_r0, q=q, real_space=True)
                        lambdas = calc_lambdas(Nx, Ny, ta, tb, new_ds, rdm_r0, derivs_set, q=q, real_space=True)
                        if e_list:
                            energies = [lambdas[i]+e_list[i] for i in range(len(lambdas))]
                            args = Nx, Ny, ta, tb, list(energies), U, q, gauge, occupations, filling, T
                        else:
                            args = Nx, Ny, ta, tb, list(np.real(lambdas)), U, q, gauge, occupations, filling, T
                    
            elif obj_func == expectation_plaq_ham:
                d_tile  = np.array(list(new_ds)*int(Ny/q)*Nx) # Need to tile because I need to keep track of entire real-space RDM0
                derivs_set = calc_derivs(ta, tb, d_tile, rdm_r0, q=q)
                lambdas = calc_lambdas(Nx, Ny, ta, tb, d_tile, rdm_r0, derivs_set, q=q)
                args = Nx, Ny, ta, tb, p, q, U, bare_occs, list(np.real(lambdas)), filling, T
                
            #if not isotropic:
            if q != 0:
                sc_min(obj_func, args, param_guess=param_guess, bounds=bounds, rdm0=rdm_r0, min_method=min_method, gtoler=gtoler, d_tol=d_tol, max_cycles=max_cycles, iteration=iteration, prev_obj=list(new_ds), mixing=mixing, glob_min=glob_min)

    else:
        print("\nSolution not found.\nPrinting scipy output and exiting:")
        print(result,"\n")
        print_time('end','gutzwiller')
        sys.exit()
