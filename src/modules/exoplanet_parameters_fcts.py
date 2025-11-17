"""
creation: 28/08/25

Contains all useful data and functions from the litterature, useful for the planet creation.

For exoplanets parameters M-P:
    means_logP_logM
    covs_logP_logM

FUTURE:
    - photoevaporation effects
    - eccentricity dispersion
"""

import numpy as np
import pandas as pd
import sys

################################################################################################
######### exoplanet distribution (M-P) #########################################################
################################################################################################
available_MP = ['', 'initial_4gaussians']

def get_gaussians_params(custom_MP_model):
    if custom_MP_model == '':
        print("No Mass-Period distribution model specified, use the default ('initial_4gaussians')")
        means_logP_logM = means_logP_logM_I4G
        covs_logP_logM = covs_logP_logM_I4G        
    elif custom_MP_model == 'initial_4gaussians':
        means_logP_logM = means_logP_logM_I4G
        covs_logP_logM = covs_logP_logM_I4G
    else:
        print("MP distribution model not existing or not implemented yet...")
        print(f"the available models are: {available_MP}")
        sys.exit("Invalid MP distribution model")

    return means_logP_logM, covs_logP_logM


################################################################################################
##### Initial 4 gaussians distribution #########################################################
################################################################################################

means_logP_logM_I4G = [[np.log10(4*10**0), np.log10(2*10**2)], # HJ
                       [np.log10(5*10**2), np.log10(7*10**2)], # CJ
                       [np.log10(1*10**1), np.log10(1*10**1)], # SE/nept
                       [np.log10(1*10**2), np.log10(1*10**0)]] # Earth-like

_sigx1_I4G, _sigy1_I4G, _rho1_I4G = 0.2, 0.4, -0.6
_sigx2_I4G, _sigy2_I4G, _rho2_I4G = 0.6, 0.6, 0
_sigx3_I4G, _sigy3_I4G, _rho3_I4G = 0.5, 0.3, 0.4
_sigx4_I4G, _sigy4_I4G, _rho4_I4G = 0.8, 0.3, 0

pg = np.array([[_sigx1_I4G, _sigy1_I4G, _rho1_I4G], 
               [_sigx2_I4G, _sigy2_I4G, _rho2_I4G], 
               [_sigx3_I4G, _sigy3_I4G, _rho3_I4G], 
               [_sigx4_I4G, _sigy4_I4G, _rho4_I4G]])

covs_logP_logM_I4G = [np.array([[_sigx1_I4G**2, _rho1_I4G*_sigx1_I4G*_sigy1_I4G], [_rho1_I4G*_sigx1_I4G*_sigy1_I4G, _sigy1_I4G**2]]),
                      np.array([[_sigx2_I4G**2, _rho2_I4G*_sigx2_I4G*_sigy2_I4G], [_rho2_I4G*_sigx2_I4G*_sigy2_I4G, _sigy2_I4G**2]]),
                      np.array([[_sigx3_I4G**2, _rho3_I4G*_sigx3_I4G*_sigy3_I4G], [_rho3_I4G*_sigx3_I4G*_sigy3_I4G, _sigy3_I4G**2]]),
                      np.array([[_sigx4_I4G**2, _rho4_I4G*_sigx4_I4G*_sigy4_I4G], [_rho4_I4G*_sigx4_I4G*_sigy4_I4G, _sigy4_I4G**2]])]



