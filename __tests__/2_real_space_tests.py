from starlight import *
import numpy as np
import sys

# Test 1: Uncorrelated nonmagnetized real-space model
print('Evaluating test 1 (B=0)...')
Nx, Ny, ta, tb, filling = 10, 10, 1, 1, 0.5
ham1 = real_ham(Nx, Ny, ta, tb)
eigvals1, eigvecs1 = np.linalg.eigh(ham1)
# Chemical potential
mu_guess1 = mu_count(eigvals1, filling)
mu_true1  = mu_calc(Nx, Ny, eigvals1, filling)
if np.round(mu_guess1,10) != np.round(mu_true1,10): # Should be true in isotropic case (by DOS symmetry)
    print('TEST FAILED: Error in chemical potential calculator. Exiting...')
    sys.exit()
# Total energy
energy_ev1 = expectation_muVT(eigvals1, mu_true1)
cutoff = int(len(eigvals1)/2)
energy_count1 = np.sum(eigvals1[0:cutoff])
if np.round(energy_ev1,10) != np.round(energy_count1,10): # Should be true in isotropic case (by eigval symmetry)
    print('TEST FAILED: Error in expectation value total energy calculator. Exiting...')
    sys.exit()
# Velocity matrices
ux1 = ux(Nx,Ny)
commutator1 = (ux1@ham1) - (ham1@ux1)
vx1_comm = (-2*np.pi/(1j))*commutator1
vx1 = vx(Nx, Ny, ta, tb)
if np.sum(vx1 - vx1_comm).round(10):
    print('TEST FAILED: Error in x velocity matrix function. Exiting...')
    sys.exit()
uy1 = uy(Nx,Ny)
commutator1 = (uy1@ham1) - (ham1@uy1)
vy1_comm = (-2*np.pi/(1j))*commutator1
vy1 = vy(Nx, Ny, ta, tb)
if np.sum(vy1 - vy1_comm).round(10):
    print('TEST FAILED: Error in y velocity matrix function. Exiting...')
    sys.exit()


# Test 2: Magnetized real-space model (anisotropic occupations/d's)
print('Evaluating test 2 (B!=0, anisotropic)...')
Nx, Ny, ta, tb, q, filling = 10, 10, 1, 1, 5, 0.5
occups4 = [0.5,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]*Nx*2
d_params4 = np.array([0.25,0.12,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]*Nx)
ham4 = real_ham(Nx, Ny, ta, tb, occupations=occups4, d_params=d_params4)
eigvals4, eigvecs4 = np.linalg.eigh(ham4)
# Chemical potential
mu_guess4 = mu_count(eigvals4, filling)
mu_true4  = mu_calc(Nx, Ny, eigvals4, filling)
if mu_guess4 == mu_true4: # Should be false in anisotropic case (by DOS symmetry)
    print('TEST FAILED: Error in chemical potential calculator. Exiting...')
    sys.exit()
# Total energy
energy_ev4 = expectation_muVT(eigvals4, mu_true4)
cutoff = int(len(eigvals4)/2)
energy_count4 = np.sum(eigvals4[0:cutoff])
if energy_ev4 == energy_count4: # Should be false in anisotropic case (by eigval symmetry)
    print('TEST FAILED: Error in expectation value total energy calculator. Exiting...')
    sys.exit()
# Velocity matrices
ux4 = ux(Nx,Ny)
commutator4 = (ux4@ham4) - (ham4@ux4)
vx4_comm = (-2*np.pi/(1j))*commutator4
vx4 = vx(Nx, Ny, ta, tb)
if np.sum(vx4 - vx4_comm).round(10):
    print('TEST FAILED: Error in x velocity matrix function. Exiting...')
    sys.exit()
uy4 = uy(Nx,Ny)
commutator4 = (uy4@ham4) - (ham4@uy4)
vy4_comm = (-2*np.pi/(1j))*commutator4
vy4 = vy(Nx, Ny, ta, tb)
if np.sum(vy4 - vy4_comm).round(10):
    print('TEST FAILED: Error in y velocity matrix function. Exiting...')
    sys.exit()


print('Real-space tests successful! Exiting...')
