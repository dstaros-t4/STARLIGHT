
##################################################################################

       _____ _______       _____  _      _____ _____ _    _ _______      _    
      / ____|__   __|/\   |  __ \| |    |_   _/ ____| |  | |__   __|  /\| |/\ 
     | (___    | |  /  \  | |__) | |      | || |  __| |__| |  | |     \ ` ' / 
      \___ \   | | / /\ \ |  _  /| |      | || | |_ |  __  |  | |    |_     _|
      ____) |  | |/ ____ \| | \ \| |____ _| || |__| | |  | |  | |     / , . \ 
     |_____/   |_/_/    \_\_|  \_\______|_____\_____|_|  |_|  |_|     \/|_|\/ 

      Sigma TrAnsveRse in LattIces with Gutzwiller-renormalized Hopping Terms
                              STARLIGHT O5003 © 2025

         Developed by: Daniel Staros, T-4, Los Alamos National Laboratory

##################################################################################

#  ABOUT

Welcome to STARLIGHT! STARLIGHT  is an all-Python code for calculating the
transverse (xy) conductivity of strongly correlated magnetized lattices. Strong
correlation is approximated via spatially unrestricted Gutzwiller renormalization 
of the hopping terms.


#  CAPABILITIES

- Automated generation of tight-binding (TB) models for magnetized, Gutzwiller 
  renormalized 2D square lattices,
- Calculation of band structures,
- Self-consistent optimization of Gutzwiller renormalization parameters.
- Real-space calculation of the Kubo transverse and longitudinal conductivity
  with or without effective correlation/Gutzwiller renormalization.


#  NOTES
- Before using STARLIGHT, and after making any changes to the source code, please 
       ensure that the tests in 'starlight/__tests__' execute smoothly (python test.py).
