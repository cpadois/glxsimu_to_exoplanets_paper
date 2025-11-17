"""
Here will be all fcts related to detectability........
"""

import numpy as np
import scipy.stats as st
from astropy import units as u

available_detectors = ['']


def TSMO03(i, P, a_R):
    """
    compute time of transit of 1 planet around its star
    INPUT:
        i [rad] orbit inclination relative to the l.o.s. plan
        P: orbital period
        a_R: semi-mayor axis of the planet orbit IN STELLAR RADIUS UNITS
    RETURN:
        transit duration (unit == unit of P)
    """
    delta_f = 2*np.arcsin(np.sqrt(1 - a_R**2 * np.cos(i)**2) / (a_R*np.sin(i)))
    Tt = P * (delta_f) / (2*np.pi)

    return Tt

def compute_detection_nb_realKeplerPl(exopl, tr_depth_col='koi_depth'):
    """
    From an exoplanet dataset, compute their transit SNR estimation by Kepler
    INPUT:
        exopl, must have the following columns:
            inclination, in degrees
            semi mayor axis, in AU
            star properties: radius [Rsun]
        tr_depth_col: 'koi_depth' or 'computed_depth'
            
    """

    print("### Initial nb of planets:", len(exopl))
    
    # convert sma in stellar radius
    a_R = exopl['koi_sma']*u.AU.to(u.Rsun) / exopl['koi_srad']
    # how many have an inclination compatible with transit detection
    calcul_tdur_allowed = (np.cos(exopl['koi_incl']*u.deg.to(u.rad)) <= 1/(a_R))

    print("### nb with incl compatible with transit det:", calcul_tdur_allowed.sum())

    # define t_dur as zero by default
    exopl['tdur_h'] = np.zeros(len(exopl))
    # compute t_dur [in hours] where allowed
    index_tdur_allowed = calcul_tdur_allowed[calcul_tdur_allowed==True].index.values
    exopl.loc[index_tdur_allowed, 'tdur_h'] = TSMO03(exopl.loc[index_tdur_allowed, 'koi_incl']*u.deg.to(u.rad),
                             exopl.loc[index_tdur_allowed, 'koi_period']*u.day.to(u.hour),
                             a_R.loc[index_tdur_allowed])
    
    print("### computing transit depth")
    # transit depth
    exopl['computed_depth'] = ((exopl.koi_prad*u.Rearth.to(u.Rsun) / exopl.koi_srad)**2)*1e6

    print("### assigning CDPP...")
    ### assign CDPP
    liste_tCDPP = np.repeat([[3, 6, 12]], len(exopl.loc[index_tdur_allowed]), axis=0)
    # compute difference for each planet (which have a tdur)
    diff = np.array([list(exopl.loc[index_tdur_allowed, 'tdur_h'].values)]).T - liste_tCDPP
    # arg of minimun difference
    arg_tCDPP = np.argmin(abs(diff), axis=1)  
    # inicialize to 0
    exopl['t_CDPP'] = np.zeros(len(exopl))
    # change arg to t_CDPP values, between 3, 6 or 12 hours
    exopl.loc[index_tdur_allowed, 't_CDPP'] = np.where(arg_tCDPP == 0, 3, np.where(arg_tCDPP == 1, 6, 12))

    ### assign them a CDPP, depending on their t_CDPP and our fit
    exopl["CDPP_N"] = np.zeros(len(exopl))

    ### first for dwarfs, 3 hours:
    index_dwarfs_3 = exopl[(exopl['koi_slogg'] > 4) & (exopl['t_CDPP'] == 3)].index
    index_dwarfs_6 = exopl[(exopl['koi_slogg'] > 4) & (exopl['t_CDPP'] == 6)].index
    index_dwarfs_12 = exopl[(exopl['koi_slogg'] > 4) & (exopl['t_CDPP'] == 12)].index

    index_giants_3 = exopl[(exopl['koi_slogg'] <= 4) & (exopl['t_CDPP'] == 3)].index
    index_giants_6 = exopl[(exopl['koi_slogg'] <= 4) & (exopl['t_CDPP'] == 6)].index
    index_giants_12 = exopl[(exopl['koi_slogg'] <= 4) & (exopl['t_CDPP'] == 12)].index

    #return exopl
    exopl = assign_CDPP_df(exopl, index_dwarfs_3, fit_dwarfs_3, disp_dwarfs_3)
    exopl = assign_CDPP_df(exopl, index_dwarfs_6, fit_dwarfs_6, disp_dwarfs_6)
    exopl = assign_CDPP_df(exopl, index_dwarfs_12, fit_dwarfs_12, disp_dwarfs_12)
    exopl = assign_CDPP_giant_df(exopl, index_giants_3.values, fit_dwarfs_3, disp_dwarfs_3)
    exopl = assign_CDPP_giant_df(exopl, index_giants_6.values, fit_dwarfs_6, disp_dwarfs_6)
    exopl = assign_CDPP_giant_df(exopl, index_giants_12.values, fit_dwarfs_12, disp_dwarfs_12)

    exopl['CDPP_eff'] = np.sqrt(exopl['t_CDPP'] / exopl['tdur_h']) * exopl['CDPP_N']

    print("### computing nb of transits")
    ### if we consider 4 yrs of missiom and a constant efficiency f0 = 92%
    exopl['N_tr'] = (4*u.yr.to(u.day) * 0.92) /exopl['koi_period']

    print("### computing SNR..")
    ### compute SNR
    exopl['SNR'] = np.sqrt(exopl['N_tr']) * exopl[tr_depth_col] / exopl['CDPP_eff']

    return exopl

def compute_detection_nb_simu(exopl, prints=True):
    """
    From an exoplanet dataset, compute their transit SNR estimation by Kepler
    INPUT:
        exopl, must have the following columns:
            inclination, in degrees
            semi mayor axis, in AU
            star properties: radius [Rsun]
        tr_depth_col: 'koi_depth' or 'computed_depth'
            
    """
    if prints:
        print("### Initial nb of planets:", len(exopl))
    
    # convert sma in stellar radius
    a_R = exopl['a_AU']*u.AU.to(u.Rsun) / exopl['radius_star']
    # how many have an inclination compatible with transit detection
    calcul_tdur_allowed = (np.cos(exopl['i_deg']*u.deg.to(u.rad)) <= 1/(a_R))

    if prints:
        print("### nb with incl compatible with transit det:", calcul_tdur_allowed.sum())

    # define t_dur as zero by default
    exopl['tdur_h'] = np.zeros(len(exopl))
    # compute t_dur [in hours] where allowed
    index_tdur_allowed = calcul_tdur_allowed[calcul_tdur_allowed==True].index.values

    exopl.loc[index_tdur_allowed, 'tdur_h'] = TSMO03(exopl.loc[index_tdur_allowed, 'i_deg']*u.deg.to(u.rad),
                             exopl.loc[index_tdur_allowed, 'period_planet']*u.day.to(u.hour),
                             a_R.loc[index_tdur_allowed])
    
    if prints:
        print("### computing transit depth")
    # transit depth
    exopl['transit_depth'] = ((exopl.radius_planet*u.Rearth.to(u.Rsun) / exopl.radius_star)**2)*1e6

    if prints:
        print("### assigning CDPP...")
    #####################
    #####################
    #####################

    ### assign CDPP (all 6hrs...!)
    exopl['t_CDPP'] = np.ones(len(exopl))*6 #np.zeros(len(exopl))
    
    ### assign them a CDPP, depending on their t_CDPP and our fit
    exopl["CDPP_N"] = np.zeros(len(exopl))

    ########### add condition that planets are "observables"
    index_dwarfs_3 = exopl[(exopl['logg'] > 4) & (exopl['t_CDPP'] == 3) & (exopl['in_st_obs'] == True)].index
    index_dwarfs_6 = exopl[(exopl['logg'] > 4) & (exopl['t_CDPP'] == 6) & (exopl['in_st_obs'] == True)].index
    index_dwarfs_12 = exopl[(exopl['logg'] > 4) & (exopl['t_CDPP'] == 12) & (exopl['in_st_obs'] == True)].index

    index_giants_3 = exopl[(exopl['logg'] <= 4) & (exopl['t_CDPP'] == 3) & (exopl['in_st_obs'] == True)].index
    index_giants_6 = exopl[(exopl['logg'] <= 4) & (exopl['t_CDPP'] == 6) & (exopl['in_st_obs'] == True)].index
    index_giants_12 = exopl[(exopl['logg'] <= 4) & (exopl['t_CDPP'] == 12) & (exopl['in_st_obs'] == True)].index

    #return exopl
    exopl = assign_CDPP_df(exopl, index_dwarfs_3, fit_dwarfs_3, disp_dwarfs_3, mag_col='Gmag')
    exopl = assign_CDPP_df(exopl, index_dwarfs_6, fit_dwarfs_6, disp_dwarfs_6, mag_col='Gmag')
    exopl = assign_CDPP_df(exopl, index_dwarfs_12, fit_dwarfs_12, disp_dwarfs_12, mag_col='Gmag')
    exopl = assign_CDPP_giant_df(exopl, index_giants_3.values, fit_dwarfs_3, disp_dwarfs_3, mag_col='Gmag')
    exopl = assign_CDPP_giant_df(exopl, index_giants_6.values, fit_dwarfs_6, disp_dwarfs_6, mag_col='Gmag')
    exopl = assign_CDPP_giant_df(exopl, index_giants_12.values, fit_dwarfs_12, disp_dwarfs_12, mag_col='Gmag')

    exopl['CDPP_eff'] = np.sqrt(exopl['t_CDPP'] / exopl['tdur_h']) * exopl['CDPP_N']

    if prints:
        print("### computing nb of transits")
    ### if we consider 4 yrs of missiom and a constant efficiency f0 = 92%
    exopl['N_tr'] = (4*u.yr.to(u.day) * 0.92) /exopl['period_planet']

    if prints:
        print("### computing SNR..")
    ### compute SNR
    exopl['SNR'] = np.sqrt(exopl['N_tr']) * exopl['transit_depth'] / exopl['CDPP_eff']

    return exopl


def compute_detection_nb_simu_custom(exopl, logg_col_name='logg'):
    """
    SAME THAT ABOVE, but allow different columns names...
    From an exoplanet dataset, compute their transit SNR estimation by Kepler
    INPUT:
        exopl, must have the following columns:
            inclination, in degrees
            semi mayor axis, in AU
            star properties: radius [Rsun]
        tr_depth_col: 'koi_depth' or 'computed_depth'
            
    """

    print("### Initial nb of planets:", len(exopl))
    
    # convert sma in stellar radius
    a_R = exopl['a_AU']*u.AU.to(u.Rsun) / exopl['radius_star']
    # how many have an inclination compatible with transit detection
    calcul_tdur_allowed = (np.cos(exopl['i_deg']*u.deg.to(u.rad)) <= 1/(a_R))

    print("### nb with incl compatible with transit det:", calcul_tdur_allowed.sum())

    # define t_dur as zero by default
    exopl['tdur_h'] = np.nan
    # compute t_dur [in hours] where allowed
    index_tdur_allowed = calcul_tdur_allowed[calcul_tdur_allowed==True].index.values
    #print(exopl.loc[index_tdur_allowed, 'period_planet']*u.day.to(u.hour))

    exopl.loc[index_tdur_allowed, 'tdur_h'] = TSMO03(exopl.loc[index_tdur_allowed, 'i_deg'].values*u.deg.to(u.rad),
                             np.array(exopl.loc[index_tdur_allowed, 'period_planet'].values*u.day.to(u.hour), dtype='float'),
                             np.array(a_R.loc[index_tdur_allowed].values, dtype='float'))
    
    print("### computing transit depth")
    # transit depth
    exopl['transit_depth'] = ((exopl.radius_planet*u.Rearth.to(u.Rsun) / exopl.radius_star)**2)*1e6

    print("### assigning CDPP...")
    #####################
    #####################
    #####################

    ### assign CDPP (all 6hrs...!)
    exopl['t_CDPP'] = np.nan
    exopl.loc[index_tdur_allowed, 't_CDPP'] = 6 #np.zeros(len(exopl))

    ### assign them a CDPP, depending on their t_CDPP and our fit
    exopl["CDPP_N"] = np.nan

    ########### add condition that planets are "observables"
    #index_dwarfs_3 = exopl[(exopl['logg'] > 4) & (exopl['t_CDPP'] == 3) & (exopl['in_st_obs'] == True)].index
    index_dwarfs_6 = exopl[(exopl[logg_col_name] > 4) & (exopl['t_CDPP'] == 6) & (exopl['in_st_obs'] == True)].index
    #index_dwarfs_12 = exopl[(exopl['logg'] > 4) & (exopl['t_CDPP'] == 12) & (exopl['in_st_obs'] == True)].index

    #index_giants_3 = exopl[(exopl['logg'] <= 4) & (exopl['t_CDPP'] == 3) & (exopl['in_st_obs'] == True)].index
    index_giants_6 = exopl[(exopl[logg_col_name] <= 4) & (exopl['t_CDPP'] == 6) & (exopl['in_st_obs'] == True)].index
    #index_giants_12 = exopl[(exopl['logg'] <= 4) & (exopl['t_CDPP'] == 12) & (exopl['in_st_obs'] == True)].index

    #return exopl
    #exopl = assign_CDPP_df(exopl, index_dwarfs_3, fit_dwarfs_3, disp_dwarfs_3, mag_col='Gmag')
    exopl = assign_CDPP_df(exopl, index_dwarfs_6, fit_dwarfs_6, disp_dwarfs_6, mag_col='Kmag')
    #exopl = assign_CDPP_df(exopl, index_dwarfs_12, fit_dwarfs_12, disp_dwarfs_12, mag_col='Gmag')
    #exopl = assign_CDPP_giant_df(exopl, index_giants_3.values, fit_dwarfs_3, disp_dwarfs_3, mag_col='Gmag')
    exopl = assign_CDPP_giant_df(exopl, index_giants_6.values, fit_dwarfs_6, disp_dwarfs_6, mag_col='Kmag')
    #exopl = assign_CDPP_giant_df(exopl, index_giants_12.values, fit_dwarfs_12, disp_dwarfs_12, mag_col='Gmag')

    exopl['CDPP_eff'] = np.sqrt(exopl['t_CDPP'] / exopl['tdur_h']) * exopl['CDPP_N']

    print("### computing nb of transits")
    ### if we consider 4 yrs of missiom and a constant efficiency f0 = 92%
    exopl['N_tr'] = ((4*u.yr.to(u.day) * 0.92) /exopl['period_planet']).astype('float')
    
    print("### computing SNR..")
    ### compute SNR
    exopl['SNR'] = np.sqrt(exopl['N_tr']) * exopl['transit_depth'] / exopl['CDPP_eff']

    return exopl



#### CDPP fit
### CDPP fit functions

## assign CDPP to dwarfs
def assign_CDPP_df(df_pl, index_pl, fct_fit, fct_sigma, mag_col='koi_kepmag', name_col_CDPP='CDPP_N'):
    """
    mag_col, name of the magnitude column, change to 'Gmag' if simu, useless if Kepler data (default)
    """
    mu = fct_fit(df_pl.loc[index_pl, mag_col])
    sig = fct_sigma(df_pl.loc[index_pl, mag_col])
    df_pl.loc[index_pl, name_col_CDPP] = st.truncnorm.rvs(a=(0-mu)/sig, b=(400-mu)/sig, loc=mu, scale=sig)

    return df_pl

## assign CDPP to giants
def assign_CDPP_giant_df(df_pl, index_pl, fct_fit_down, disp_fit_down, mag_col='koi_kepmag', name_col_CDPP='CDPP_N'):
    up_or_down = np.random.uniform(0, 1, len(index_pl))

    mu = np.where(up_or_down>=0.5, fit_giants_6(df_pl.loc[index_pl, mag_col]), fct_fit_down(df_pl.loc[index_pl, mag_col]))
    sig = np.where(up_or_down>=0.5, disp_giants_up(df_pl.loc[index_pl, mag_col]), disp_fit_down(df_pl.loc[index_pl, mag_col]))
    #return mu, sig
    df_pl.loc[index_pl, name_col_CDPP] = st.truncnorm.rvs(a=(0-mu)/sig, b=(400-mu)/sig, loc=mu, scale=sig)

    return df_pl

## actual fit to measured CDPP
# dwarf, 3h
def fit_dwarfs_3(x):
    return 0.01 * 2**(x-0.9) + 14 #fit middle

# dwarf, 6h
def fit_dwarfs_6(x):
    return 0.01 * 2**(x-1.5) + 14 #fit middle
         # 0.01 * 2**(x-1.7) + 15 #better fit?

# giants up, 6h (same for 3 and 12h)
def fit_giants_6(x):
    return 0.08 * 1.78**(x-0.8) + 55 #fit middle
         # 0.01 * 1.8**(x+2.4) +55 #better fit?

# dwarf, 12h
def fit_dwarfs_12(x):
    return 0.01 * 2**(x-2) + 14 #fit middle

## fit dispersion
def disp_giants_up(x):
    return 0.08 * 1.78**x * (1.78**(-0.6) - 1.78**(-0.8)) #10

## fit disp dward AND giants down
def disp_dwarfs_3(x):
    return 0.01 * 2**x * (2**(-0.7) - 2**(-0.9))

def disp_dwarfs_6(x):
    return 0.01 * 2**x * (2**(-1.3) - 2**(-1.5)) #4*x-40

def disp_dwarfs_12(x):
    return 0.01 * 2**x * (2**(-1.8) - 2**(-2))