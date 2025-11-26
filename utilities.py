"""
  Filename: utilities.py
    Author: Daniel Staros
   Summary: Contains utility functions which are used throughout STARLIGHT to enhance I/O functionality.
   Imports: datetime.datetime
"""

from datetime import datetime

def print_time(instance,process):
    """ Function which prints the date and time at the beginning and end of a STARLIGHT (1) Gutzwiller optimization, (2) 
        conductivity calculation, or (3) band structure calculation.
    """
    time = datetime.now()
    if process == 'gutzwiller':
        time_var = 'self-consistent optimization of Gutzwiller parameters (self_consist.sc_min)'
    elif process == 'conductivity':
        time_var = 'conductivity calculation (observables.sigma_xy_range/sigma_longitud_range)'
    elif process == 'bands':
        time_var = 'band structure calculation (observables.calc_bands)'

    if instance == 'start':
        print(f"STARLIGHT✱ {time_var} started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif instance == 'end':
        print(f"STARLIGHT✱ {time_var} finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

