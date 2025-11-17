"""
Useful functions for the creation of a stellar population from stellar particles of a galactic simulation

Contains:
    - IMF function(s)
    - stellar multiplicity and companion frequency functions
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import sys
from astropy.table import Table
import galpy.util.coords


################################################################################################
######### IMF fcts #############################################################################
################################################################################################

available_imf = ['kirkpatrick24']

def choice_imf(custom_imf):
    if custom_imf == 'kirkpatrick24':
        imf_object = imf_pdf_kirkpatrick(a=0.1, b=100)
    else:
        print("IMF model not existing or not implemented yet")

    return imf_object


### IMF fcts
def kirkpatrick_indiv(m, alpha):
    return m**alpha

def IMF_kirkpatrick_norm(m_array, norm=False):
    ### normalized in order to have IMF(0.1) = 1 (lowest value of our 0.1-1.6 Msun range)
    ### allow m_array or single value
    a1 = 2.3
    a2 = 1.3
    a3 = 0.25
    a4 = 0.6
    
    c1 = 0.0150
    c2 = 0.0273
    c3 = 0.134
    c4 = 0.0469

    # normalization coefficients
    if norm:
        norm3 = kirkpatrick_indiv(0.1, -a3)
        norm4 = kirkpatrick_indiv(0.05, -a4)/(kirkpatrick_indiv(0.05, -a3)/norm3)
        norm2 = kirkpatrick_indiv(0.22, -a2)/(kirkpatrick_indiv(0.22, -a3)/norm3)
        norm1 = kirkpatrick_indiv(0.55, -a1)/(kirkpatrick_indiv(0.55, -a2)/norm2)
    else: 
        norm1, norm2, norm3, norm4 = (1/c1, 1/c2, 1/c3, 1/c4)
    
    try:
        res = np.ones(len(m_array))
        for i in range(len(m_array)):
            m = m_array[i]
            if m < 0.05:
                res[i] = kirkpatrick_indiv(m, -a4) / norm4
            elif m < 0.22:
                res[i] = kirkpatrick_indiv(m, -a3) / norm3
            elif m < 0.55:
                res[i] = kirkpatrick_indiv(m, -a2) / norm2
            else:
                res[i] = kirkpatrick_indiv(m, -a1) / norm1
    except:
        ### single value
        m = m_array
        if m < 0.05:
            res = kirkpatrick_indiv(m, -a4) / norm4
        elif m < 0.22:
            res = kirkpatrick_indiv(m, -a3) / norm3
        elif m < 0.55:
            res = kirkpatrick_indiv(m, -a2) / norm2
        else:
            res = kirkpatrick_indiv(m, -a1) / norm1
    ### divide by integral of the fct, previously calculated
    return res/0.08513892289372894

def cdf_kirkpatrick(m_array, norm=True):
    c1 = 0.0150
    a1 = 2.3
    c2 = 0.0273
    a2 = 1.3
    c3 = 0.134
    a3 = 0.25
    c4 = 0.0469
    a4 = 0.6
    
    try:
        res = np.ones(len(m_array))
        for i in range(len(m_array)):
            m = m_array[i]
            if m < 0.1:
                res[i] = 0
            elif m < 0.22:
                res[i] = (kirkpatrick_indiv(m, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3)
            elif m < 0.55:
                res[i] = (kirkpatrick_indiv(0.22, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3) \
                        + (kirkpatrick_indiv(m, -a2+1)-kirkpatrick_indiv(0.22, -a2+1)) *c2/(1-a2)
            else:
                res[i] = (kirkpatrick_indiv(0.22, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3) \
                        + (kirkpatrick_indiv(0.55, -a2+1)-kirkpatrick_indiv(0.22, -a2+1)) *c2/(1-a2) \
                        + (kirkpatrick_indiv(m, -a1+1)-kirkpatrick_indiv(0.55, -a1+1)) *c1/(1-a1)
    except:
        ### single value
        m = m_array
        if m < 0.1:
            res = 0
        elif m < 0.22:
            res = (kirkpatrick_indiv(m, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3)
        elif m < 0.55:
            res = (kirkpatrick_indiv(0.22, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3) \
                + (kirkpatrick_indiv(m, -a2+1)-kirkpatrick_indiv(0.22, -a2+1)) *c2/(1-a2)
        else:
            res = (kirkpatrick_indiv(0.22, -a3+1)-kirkpatrick_indiv(0.1, -a3+1)) *c3/(1-a3) \
                + (kirkpatrick_indiv(0.55, -a2+1)-kirkpatrick_indiv(0.22, -a2+1)) *c2/(1-a2) \
                + (kirkpatrick_indiv(m, -a1+1)-kirkpatrick_indiv(0.55, -a1+1)) *c1/(1-a1)

    ### normalization coefficient previously calculated 
    if norm:
        coeff_norm = 0.08513892289372894
    else:
        coeff_norm = 1
    return res/coeff_norm


### def distribution class
class imf_pdf_kirkpatrick(st.rv_continuous):
    def _pdf(self, x):
        return IMF_kirkpatrick_norm(x)
    def _cdf(self, x):
        return cdf_kirkpatrick(x)


################################################################################################
######### stellar multiplicity and nb of companions ############################################
################################################################################################

###### multiplicity fraction
### values from `Table 1` of Offner+23 (just the ones appearing in fig.1) 
offner23_mf = np.array([[0.019, 0.058, 8, 6], [0.05, 0.08, 15, 4], #[0.08, 0.095, 19, 7], [0.06, 0.15, 20, 4],
                        [0.075, 0.15, 19, 3], [0.15, 0.3, 23, 2], [0.3, 0.6, 30, 2], [0.75, 1.25, 46, 3],
                        [0.75, 1, 42, 3], [1.0, 1.25, 50, 4], [1.6, 2.4, 68, 7], #[0.85, 1.5, 47, 3], [1.6, 2.4, 68, 7],
                        [3.0, 5.0, 81, 6], [5.0, 8.0, 89, 5], [8.0, 17.0, 93, 4], [17.0, 50.0, 96, 4] ])

### create df
df_mf = pd.DataFrame(offner23_mf, columns=["mass_min", "mass_max", "MF_value", "MF_err"])
### create column for "mean" mass, mean in log scale (to be consistent with fig.1 of Offner+23)
df_mf['mass_mean'] = 10**((np.log10(df_mf['mass_max'])+np.log10(df_mf['mass_min']))/2.)

### calculate the "error" of the mass, not really the error but to cover all mass range from mass_min to mass_max
errors_mass = pd.concat([abs(df_mf['mass_min'] - df_mf['mass_mean']).T, abs(df_mf['mass_max'] - df_mf['mass_mean']).T], axis=1)

### we keep only points in our mass range, so the fitting is more easy (~linear in natural scale (not log))
m_selection = (df_mf['mass_mean']>=0.1) & (df_mf['mass_mean']<=1.6)

### linear fit
polyfit_MF = np.polyfit(df_mf[m_selection]['mass_mean'], df_mf[m_selection]['MF_value'], deg=2)
fit_MF = np.poly1d(polyfit_MF)

def multiplicity_fraction(mass):
    return fit_MF(mass)

###### nb of companions
offner23_cf = np.array([[0.019, 0.058, 0.08, 0.06], [0.05, 0.08, 0.16, 0.04], [0.08, 0.095, 0.19, 0.07], #[0.06, 0.15, 0.2, 0.04],
                        [0.075, 0.15, 0.21, 0.03], [0.15, 0.3, 0.27, 0.03], [0.3, 0.6, 0.38, 0.3], [0.75, 1.25, 0.6, 0.04],
                        [0.85, 1.5, 0.62, 0.04], [1.6, 2.4, 0.99, 0.13],
                        [3.0, 5.0, 1.28, 0.17], [5.0, 8.0, 1.55, 0.24], [8.0, 17.0, 1.8, 0.3], [17.0, 50.0, 2.1, 0.3] ])

### create df
df_cf = pd.DataFrame(offner23_cf, columns=["mass_min", "mass_max", "CF_value", "CF_err"])
### create column for "mean" mass, mean in log scale (to be consistent with fig.1 of Offner+23)
df_cf['mass_mean'] = 10**((np.log10(df_cf['mass_max'])+np.log10(df_cf['mass_min']))/2.)

### calculate the "error" of the mass, not really the error but to cover all mass range from mass_min to mass_max
errors_mass = pd.concat([abs(df_cf['mass_min'] - df_cf['mass_mean']).T, abs(df_cf['mass_max'] - df_cf['mass_mean']).T], axis=1)

### we keep only points in our mass range, so the fitting is more easy (~linear in natural scale (not log))
m_selection = (df_cf['mass_mean']>=0.1) & (df_cf['mass_mean']<=1.6)

### linear fit
polyfit_CF = np.polyfit(df_cf[m_selection]['mass_mean'], df_cf[m_selection]['CF_value'], deg=1)
fit_CF = np.poly1d(polyfit_CF)

def companion_frequency(mass):
    return fit_CF(mass)


################################################################################################
######### selection the stellar particles in the galactic simulation ###########################
################################################################################################

dict_preselected_regions_params = {
    'SN': ['box', 7.7, 8.7, -0.5, 0.5, 175, 185],
    'inner': ['box', 3.8, 4.2, -0.3, 0.3, 175, 185],
    'outer': ['box', 14, 16, -1.0, 1.0, 175, 185],
    'upper': ['box', 7.5, 8.9, 2.5, 4.0, 165, 195],
    'lower': ['box', 7.5, 8.9, -4.0, -2.5, 165, 195],
    'phi60': ['box', 7.7, 8.7, -0.5, 0.5, 55, 65],
    'phi300': ['box', 7.7, 8.7, -0.5, 0.5, 295, 305],
    'center': ['cyl', 0.07, 0.07],
    'test': ['box', 7.9, 8.1, -0.1, 0.1, 179, 181]
}

def _get_simu_file_name(simu_name: str) -> str:
    if simu_name.endswith('696e11'):
        simu_file_name = '6.96e11.01024_stars_rsphmax25.0.fits'
    elif simu_name.endswith('708e11'):
        simu_file_name = '7.08e11.01024_stars_rsphmax25.0.fits'
    elif simu_name.endswith('755e11'):
        simu_file_name = '7.55e11.01024_stars_rsphmax25.0.fits'
    elif simu_name.endswith('826e11'):
        simu_file_name = '8.26e11.02000_stars_rsphmax25.0.fits'
    elif simu_name.endswith('112e12'):
        simu_file_name = '1.12e12.02000_stars_rsphmax25.0.fits'
    elif simu_name.endswith('279e12'):
        simu_file_name = '2.79e12.02000_stars_rsphmax25.0.fits'
    else:
        print("Simulation name not supported (yet), the available are:")
        print("'NIHAO_UHD_g696e11', 'NIHAO_UHD_g696e11', 'NIHAO_UHD_g708e11', 'NIHAO_UHD_g755e11', 'NIHAO_UHD_826e11', 'NIHAO_UHD_g112e12', 'NIHAO_UHD_g279e12'")
        sys.exit("Invalid simulation name")
    return simu_file_name

def _change_galactic_coordinates(table_simu: Table) -> Table:
    """
    make a serie of changes in coordinates
    """
    # rotate the frame by -60 deg and flip the angular-momentum vector
    rotation = -60.

    table_simu['x_new'] = -table_simu['x'] * np.cos(np.deg2rad(rotation)) + table_simu['y'] * np.sin(np.deg2rad(rotation))
    table_simu['y_new'] = +table_simu['x'] * np.sin(np.deg2rad(rotation)) + table_simu['y'] * np.cos(np.deg2rad(rotation))
    table_simu['phi_new'] = np.rad2deg(np.arctan2(table_simu['x_new'], table_simu['y_new'])) + 180.

    # For convenience, also compute r & rbirth
    table_simu['rbirth'] = np.sqrt(table_simu['xb']**2.+ table_simu['yb']**2)
    table_simu['r'] = np.sqrt(table_simu['x']**2. + table_simu['y']**2)
    # Get vphi from the Cartesian values
    table_simu['vphi'] = (table_simu['x']*table_simu['vy']-table_simu['y']*table_simu['vx']) / np.sqrt(table_simu['x']**2.+table_simu['y']**2)

    rotgal = +30.
    table_simu['x_gal'] = +table_simu['x'] * np.cos(np.deg2rad(rotgal)) - table_simu['y'] * np.sin(np.deg2rad(rotgal))
    table_simu['y_gal'] = +table_simu['x'] * np.sin(np.deg2rad(rotgal)) + table_simu['y'] * np.cos(np.deg2rad(rotgal))
    table_simu['phi_gal'] = np.rad2deg(np.arctan2(-table_simu['y_gal'], table_simu['x_gal']))

    ### convert XYZ gal to XYZ centered on the Sun...
    conversion = galpy.util.coords.galcenrect_to_XYZ(table_simu['x_gal'], table_simu['y_gal'], table_simu['z'], Xsun=-8.2, Zsun=0.05)

    table_simu['x_sun'] = conversion[:, 0]
    table_simu['y_sun'] = conversion[:, 1]
    table_simu['z_sun'] = conversion[:, 2]

    ### convert XYZ centered on the Sun to lbd
    conversion = galpy.util.coords.XYZ_to_lbd(table_simu['x_sun'], table_simu['y_sun'], table_simu['z_sun'])

    table_simu['l'] = conversion[:,0]
    table_simu['b'] = conversion[:,1]
    table_simu['dist'] = conversion[:,2]
    #print(conversion)

    ### convert l, b to ra/dec
    conversion = galpy.util.coords.lb_to_radec(table_simu['l'], table_simu['b'])
    table_simu['ra'] = conversion [:,0]
    table_simu['dec'] = conversion [:,1]


    ### distance estimation
    table_simu['dist_sun'] = np.sqrt((table_simu['x_gal']+8.2)**2 + (table_simu['y_gal'])**2 + (table_simu['z']-0.05)**2)

    return table_simu

def sp_selection(table_simu: Table,
                  sp_selection_type: str,
                  params_sp_sel: list) -> pd.DataFrame:
    if sp_selection_type == 'box':
        # params_sp_sel format: Rmin, Rmax, zmin, zmax, phimin, phimax
        rmin, rmax, zmin, zmax, phimin, phimax = params_sp_sel
        cut = table_simu[(table_simu['r'] > rmin) & (table_simu['r'] < rmax)
                         & (table_simu['z'] > zmin) & (table_simu['z'] < zmax)
                         & (table_simu['phi_new'] > phimin) & (table_simu['phi_new'] < phimax)]
    elif sp_selection_type == 'cyl':
        # params_sp_sel format: Rmax, abs(z)_max
        rmax, zmax = params_sp_sel
        cut = table_simu[(table_simu['r'] < rmax) & (abs(table_simu['z']) < zmax)]

    return cut

def select_stellar_particles(params):
    """
    open the galactic simulation and select the stellar particles corresponding to the asked region

    available simulations: 
    regions: 
        - pre selected regions (Padois+25):
        - custom region: [FORMAT, list_of_corresponding_parameters] 
            available formats:
             name  | corresponding parameters necessary     | description/units
            -----------------------------------------------------------------------------------------
             'box' | Rmin, Rmax, zmin, zmax, phimin, phimax | Rmxx and zmxx in kpc, phimxx in degrees
             'obs' | ra_center, dec_center, fov_width, dist_max
    """
    path_simu_file = params['INPUT_STARS']['path_galactic_simu'] + _get_simu_file_name(params['INPUT_STARS']['galactic_simu'])
    simu = Table.read(path_simu_file)
    simu = _change_galactic_coordinates(simu)

    ### read region selection from params

    ### if pre selected region: retreive the corresponding cuts
    if params['INPUT_STARS']['galactic_region'] in dict_preselected_regions_params.keys():
        print(f"{params['INPUT_STARS']['galactic_region']} region selected")
        sp_selection_type = dict_preselected_regions_params[params['INPUT_STARS']['galactic_region']][0]
        params_sp_selection = dict_preselected_regions_params[params['INPUT_STARS']['galactic_region']][1:]
    else: ### if custom selection, check if correct format
        print("Selection of a custom region: not available yet...")
        sys.exit("Invalid region selected")
    ### make the cuts
    sp = sp_selection(simu, sp_selection_type, params_sp_selection)
    print(f"{len(sp)} particles selected")
    if len(sp) > 1300:
        print("Warning, may take (several?) hours...")
        check_proceed = input("Do you still want to proceed? [y/n]")
        if not check_proceed:
            sys.exit("User interruption")

    return sp

