from starlight import *
import numpy as np
import sys

# Test 1: General square lattice generator test (Nx != Ny)
print('Evaluating test 1: Lattice generator...')
Nx, Ny = 8, 10
square_lattice_10 = site_list(Nx, Ny)

if not (Nx*Ny == len(square_lattice_10)):
    print('TEST FAILED: Error in lattice site list generator')
    sys.exit()

# Fermi function test
print('Evaluating test 2: Fermi function...')
mu  = 0
energy_list = np.arange(-2, 2.1, 0.1)
fd_list     = [fermi_func(energy, mu) for energy in energy_list]
plt.plot(energy_list, fd_list)
plt.title('Test: Fermi-Dirac Function')
plt.xlabel('Energy'); plt.ylabel(r'f($\mu$=0,E)')
plt.show()

# Gutzwiller function test
print('Evaluating test 1: Gutzwiller renormalization...')
n = 0.5
d_list     = np.arange(0,0.2525,0.025)
alpha_list = [sq_a(n,n,d)**2 for d in d_list]
if any(alpha.imag != 0 for alpha in alpha_list):
    print('TEST FAILED: Error in Gutzwiller function definition')
    sys.exit()
else:
    alpha_list = [alpha.real for alpha in alpha_list]
    plt.plot(d_list, alpha_list)
    plt.title('Test: Gutzwiller Renormalization Function')
    plt.xlabel('d'); plt.ylabel(r'$\alpha$($n_s=0.5$,$n_{sp}=0.5$,d)')
    plt.show()

print('Simple tests successful! Exiting...')
