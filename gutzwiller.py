"""
  Filename: gutzwiller.py
    Author: Daniel Staros
   Summary: Contains accessory functions for calculating Gutzwiller renormalization parameters from occupations and double 
            occupancy parameters.
   Imports: cmath, numpy, sympy
"""

import cmath
import numpy as np
import sympy as sp

def sq_a(ns, nsbar, d):
    """ Calculates the nonpolarized Gutzwiller renormalization parameter of site j, sqrt{alpha_j}, from 
        the corresponding average occupations and double occupancy parameter.
      --parameters--
          ns: Average occupation <n_{js}> = <c_{js}^{dagger} c_{js}>
       nsbar: Average occupation <n_{js'}> = <c_{js'}^{dagger} c_{js'}> (s' = opposite spin to s)
           d: Double occupancy parameter d_j
    """
    n = ns+nsbar
    if 0 <= d <= ((n**2)/4): # Necessary for physically allowed range of sq_a
        frac1 = ((ns-d)*(1-n+d)) / (ns*(1-ns))
        frac2 = (d*(nsbar-d)) / (ns*(1-ns))
        sq_a = cmath.sqrt(frac1)+cmath.sqrt(frac2)
    else:          # ...in such cases, set corresponding alpha to 0 to discourage unphysical d moves.
        sq_a = 0

    return sq_a

def sq_a_sp():
    """ Defines symbolic Gutzwiller renormalization parameter
      --parameters--
          ns: Average occupation <n_{js}> = <c_{js}^{dagger} c_{js}>
       nsbar: Average occupation <n_{js'}> = <c_{js'}^{dagger} c_{js'}> (s' = opposite spin to s)
           d: Double occupancy parameter d_j
    """
    ns, nsbar, d = sp.symbols('n_s, n_sbar, d', real=True)
    frac1 = ((ns-d)*(1-(ns+nsbar)+d)) / (ns*(1-ns))
    frac2 = (d*(nsbar-d)) / (ns*(1-ns))
    sq_a_sp  = sp.sqrt(frac1) + sp.sqrt(frac2)

    return sq_a_sp

def deriv_sq_a_sp(wrt, ns_sub=False, nsbar_sub=False, d_sub=False):
    """ Defines symbolic derivative of Gutzwiller renormalization parameter,
        and offers an option for numerical substitution.
      --parameters--
         wrt: ``with respect to" - derivative taken w.r.t this variable
    """
    ns, nsbar, d = sp.symbols('n_s, n_sbar, d', real=True)
    if wrt == 'n_s':
        deriv_sq_a = sp.diff(sq_a_sp(), ns)
    elif wrt == 'n_sbar':
        deriv_sq_a = sp.diff(sq_a_sp(), nsbar)
    elif wrt == 'd':
        deriv_sq_a = sp.diff(sq_a_sp(), d)
    else:
        raise ValueError("Input argument must be set to either 'n_s', 'n_sbar' or 'd'.")

    if (ns_sub) and (not d_sub):
        if 0 < ns_sub < 1: # Cannot equal 0 or 1 because alpha would diverge
            deriv_sq_a = deriv_sq_a.subs({ns:ns_sub})
        else:
            raise ValueError("ns_sub must be between (0.0, 1.0).")
    if (nsbar_sub) and (not d_sub):
        if 0 < nsbar_sub < 1: # Could in principle equal 0 or 1, but need n_s + n_sbar = n, so can't.
            deriv_sq_a = deriv_sq_a.subs({nsbar:nsbar_sub})
        else:
            raise ValueError("ns_sub must be between (0.0, 1.0).")
    if (d_sub) and (not ns_sub) and (not nsbar_sub):
        if 0 <= d_sub <= 1:
            deriv_sq_a = deriv_sq_a.subs({d:d_sub})
        else:
            raise ValueError("d_sub must be between [0.0, 1.0].")

    if d_sub and ns_sub and nsbar_sub:
        d_sub = np.abs(d_sub)
        n_sub = ns_sub + nsbar_sub
        if 0 <= d_sub <= ((n_sub**2)/4):
            deriv_sq_a = deriv_sq_a.subs({ns:ns_sub, nsbar:nsbar_sub, d:d_sub})
        else:
            deriv_sq_a = 0.0

    return deriv_sq_a, {'n_s': ns, 'n_sbar':nsbar, 'd': d }
