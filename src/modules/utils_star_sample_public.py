"""
28/08/25: copied from main code
"""

import os.path
import sys
import numpy as np

from astropy import units as u
import astropy.table

import pandas as pd
from astropy.table import Table

from scipy import integrate
import scipy.stats as st

import astropy.constants as cst

from tqdm import tqdm
from timeit import default_timer as timer

import matplotlib.pyplot as plt

#from occ_rates_fcts import (multiplicity_Earthlike, multiplicity_SE_nept, multiplicity_giants,
#                            occ_rate_Earthlike_MFeh, occ_rate_SEnept_MFeh, occ_rate_giants_MFeh)
#from external_datas import (means_logP_logM, covs_logP_logM, imf_pdf_kirkpatrick,
#                            multiplicity_fraction, companion_frequency)
#from init_custom import (extra_stellar_params, directory_to_save_files,
#                         sample_name, sample_date, extra_info_in_filename,
#                         column_stellar_mass, column_stellar_metallicity,
#                         column_teff, column_luminosity, column_stellar_radius)

from src.modules import occ_rates_fcts as OCC_RATE_MODULE
from src.modules import exoplanet_parameters_fcts as EXOPL_MODULE
from src.modules import glxsimu_to_stars_fcts as SIMU_STARS_MODULE
from src.modules import read_params
#import init_custom as INIT_MODULE


################################################################################################
######### StarSample class definition ##########################################################
################################################################################################


class StarSample:
    """
    Create a DataFrame of stars, organise them in multiple stellar systems etc.
    >>> create only if need to generate stars from simu (only NIHAO-UHD for now)
    >>> if only need to create planet population, create PlanetSample object (see below)
    """
    def __init__(self):
        """
        INPUT:
            no input, use _param_new_init() instead
            use ALONE only if star creation already done AND saved into a parquet file...
            if new initialisation, use _param_new_init after
        
        FOR STAR CREATION:
            ...
        FOR PLANET CREATION:

        """
        ### data file name
        self.file_name = ''
        self.file_date = ''
        self.extra_info_file = ''
        self.path_file_directory = ''
        # df with star particleS param, 1 line per particle
        self.star_particules = None
        self.isochrones = None
        # def col df final
        #tot_columns = np.append(star_particules.columns, np.append(isochrones.columns, 'mass_star_rvs'))
        # where to store the df with all stars
        self.df = None
        self.singles = None
        self.binaries = None
        self.multiples = None
        #self.planets = None
        #self.planetary_systems = None
        # other useful params
        self.imf = None
        self.imf_mean_mass = None
        # pr stocker trucs à vérifier...
        self.test = None
        self.test1 = None
        self.test2 = None
        self.nb_tot_binary_syst = 0
        self.sigma_gaussian_e = 0.4
        # to avoid loading into memory all useless df
        self.load_general_df = False
        self.load_singles_df = True
        self.load_binaries_df = False
        self.load_multiples_df = False
        self.load_planets_df = False
        ##### default column names: mandatory mass and metallicity columns
        self.feh_col_name = ''
        self.mass_col_name = ''
        # extra possible columns (less output if not given)
        self.teff_col_name = ''
        self.logg_col_name = False
        self.radius_star_col_name = ''
        self.lum_col_name = ''
        self.age_col_name = False
        self.extra_stellar_params = False
        
    def param_new_init(self, params, star_particules):
        """
        INPUT:
            star_particules: df with stellar particles parameters, 1 line per bin
            isochrones: df with all isochrones
            imf: object of the class IMF_Kirkpatrick
        """
        # df with star particleS param, 1 line per particle
        self.star_particules = star_particules
        # df with all isochrones
        self.isochrones = Table.read(params['INPUT_STARS']['path_isochrones_file']).to_pandas()
        self.isochrones.rename(columns={'z':'z_pc', 'age':'age_pc'}, inplace=True)
        # def col df final
        tot_columns = np.append(star_particules.columns, np.append(self.isochrones.columns, 'mass_star_rvs'))
        # where to store the df with all stars
        self.df = pd.DataFrame(columns = tot_columns)
        # other useful params
        custom_imf = SIMU_STARS_MODULE.choice_imf(params['INPUT_STARS']['imf'])
        self.imf = custom_imf
        self.imf_mean_mass = custom_imf.mean() #to not calculate it again every time

    def extract_column_names(self, config):
        """
        extract columns' names from the given config file and save them as attributes
        config: either a dict with the column names or directly the configfile (see format example **somewhere**) 
        """
        # if config is a config file, read it to dictionary
        if not isinstance(config, dict):
            config = read_params.get_columns(config)
        # assign column names to class attributes
        self.mass_col_name = config['COLUMNS']['column_stellar_mass']
        self.feh_col_name = config['COLUMNS']['column_stellar_metallicity']
        self.teff_col_name = config['COLUMNS']['column_teff']
        self.radius_star_col_name = config['COLUMNS']['column_stellar_radius']
        self.lum_col_name = config['COLUMNS']['column_luminosity']
        self.extra_stellar_params = config['COLUMNS']['extra_stellar_params']
            

    def _get_filename_and_path(self, 
                               which_table: str, 
                               params: dict)-> str:
        """
        return path + filename for a given table 
        to load or save it

        available tables:
            - [df] ([file identificator]): [str to input in 'which_table']
            - self.singles ('_SINGLES_'): singles
            - self.planets ('_PLANETS_'): planets
            - self.binaries ('_BINARIES_'): binaries
            - self.multiples ('_MULTIPLES_'): multiples
            - self.df ('_ALLSTARS_'): allstars (only from simulation)
        """
        str_to_filecode = {'singles': '_SINGLES_',
                           'binaries': '_BINARIES_',
                           'multiples': '_MULTIPLES_',
                           'allstars': '_ALLSTARS_'}
    
        filename = params['OUTPUT']['sample_name'] + str_to_filecode[which_table] \
            + params['OUTPUT']['date'] + params['OUTPUT']['extra_info_in_filename'] + '.parquet'
        return params['OUTPUT']['path'] + filename

    def _associate_particules_to_iso(self):
        """
        for each stellar particle, find isochrone with closest age and metallicity from isochrone table
        """
        # make a list with all ages and metallicities available in the isochrone table
        iso_mh = self.isochrones["MH"].unique()
        iso_logage = self.isochrones["logage"].unique()-9 # convert in log(Gyr)

        # for each stellar bin, find the better couple of values
        diff_mh_index = np.zeros(len(self.star_particules), dtype=np.int32)
        diff_age_index = np.zeros(len(self.star_particules), dtype=np.int32)
        tab_sp = Table.from_pandas(self.star_particules)
        tab_iso = Table.from_pandas(self.isochrones)

        for i in tqdm(range(len(self.star_particules))):
            # compare age and met with all values in iso table
            diff_mh = np.sqrt((tab_sp[i]["feh"] - iso_mh)**2)
            diff_age = np.sqrt((np.log10(tab_sp[i]["age"]) - iso_logage)**2)

            # save index of the minimum diff
            diff_mh_index[i] = np.argmin(diff_mh)
            diff_age_index[i] = np.argmin(diff_age)

        # create columns with all matched values of age and met
        assoc_mh = pd.DataFrame(iso_mh[diff_mh_index], columns=["mh_iso"], dtype=float)
        assoc_age = pd.DataFrame(iso_logage[diff_age_index], columns=["logage_iso"], dtype=float)

        # add to stellar particules param:
        self.star_particules = pd.concat([self.star_particules, assoc_mh, assoc_age], axis=1)

        ### now save to the table the minimal and maximal masses allowed in each isochrone
        m_min = np.zeros(len(self.star_particules))
        m_max = np.zeros(len(self.star_particules))
        for i in tqdm(range(len(self.star_particules))):
            select_iso = self.isochrones[(self.isochrones["MH"]==self.star_particules.loc[i, "mh_iso"]) 
                                         & (self.isochrones["logage"]==self.star_particules.loc[i, "logage_iso"]+9)]
            # for each stellar particule, take min and maximum mass values in the associated isochorne
            m_min[i] = select_iso["m_ini"].min()
            m_max[i] = select_iso["m_ini"].max()

        # store in df
        self.star_particules["m_min_iso"] = m_min
        self.star_particules["m_max_iso"] = m_max


    def _creation_stars_1particle(self, star_bin):
        """
        creation of stars for 1 PARTICLE
        INPUT:
            star_bin: df line w/ parameters of 1 stellar bin
        RETURN:
            df with created stars (from 1 stellar bin)
        """
        
        ### where to save stars from this bin
        df_bin = pd.DataFrame(columns = self.df.columns)
        
        # select the isochrone corresponding to age & metallicity of the stellar on
        select_iso = self.isochrones[(self.isochrones["MH"]==star_bin["mh_iso"]) & (self.isochrones["logage"]==star_bin["logage_iso"]+9)]
        
        # if m_max > 1.6, take 1.6 as the upper bound
        #m_min_iso = star_bin["m_min_iso"]
        if star_bin["m_max_iso"] > 1.6:
            m_max_iso = 1.6
        else:
            m_max_iso = star_bin["m_max_iso"]
        
        ### random sampling of stellar mass following Kirkpatrick+23 IMF (between 0.1 and 100 M_sun)
        size = star_bin['mass']/self.imf_mean_mass
        
        mass_sample = self.imf.rvs(size=int(size))
        
        ### keep only masses between 0.1 M_sun and M_max_iso or 1.6 M_sun
        mass_sample_keep = mass_sample[mass_sample<m_max_iso]
        # sort them in descending order, to create binary syst after...
        mass_sample_sorted = np.flip(np.sort(mass_sample_keep))
    
        ### convert series into a df
        star_bin = star_bin.to_frame().T
        
        ### put masses in df...
        df_bin[self.mass_col_name] = mass_sample_sorted

        ### ...and bin parameters (all the same because same bin!)
        df_bin.loc[range(len(mass_sample_sorted)), star_bin.columns] = star_bin.iloc[0].values
        
        ### for each mass value, match to the closest isochrone's point
        for j in range(len(mass_sample_sorted)):
            
            # find closest isochrone point
            distance_ss = np.sqrt((mass_sample_sorted[j] - select_iso["m_ini"])**2.) 
            iso_point = select_iso.iloc[np.argmin(distance_ss)]
            
            # create df for this mass and fill all param
            if j == 0:
                liste_iso = np.array([iso_point])
            else: 
                liste_iso = np.append(liste_iso, [iso_point], axis=0)

        ### iso param to df:
        df_bin.loc[range(len(df_bin)), select_iso.columns] = pd.DataFrame(liste_iso, columns=select_iso.columns)
        
        return df_bin

    def creation_stars(self, params, path_saving_stars=None, extraextrainfo=''):
        """
        create stars for the stellar particles selected and input of the current instance of the class
        save them in a parquet file (if not already existing)
        
        """
        if path_saving_stars is None:
            path_saving_stars = self._get_filename_and_path('allstars', params)
        
        if os.path.exists(path_saving_stars):
            if self.load_general_df:
                self.df = pd.read_parquet(path_saving_stars)
                print("StarSample.df loaded")
            else:
                print("StarSample.df already exists but will not be loaded into raw memory")
        else:
            ### associate each particle to the closest isochrone (in age and metallicity)
            self._associate_particules_to_iso()
            #print(self.star_particules)
            ### creation of a new df to contain all created stars
            ### for each stellar particule, create stars from previously associated isochrone
            print("Start creating stars from stellar particles (takes around 15 sec per particle...)")
            for i in tqdm(range(len(self.star_particules))):
                self.df = pd.concat([self.df, self._creation_stars_1particle(self.star_particules.loc[i])], ignore_index=True)
            ### save them to fits --> parquet
            #Table.from_pandas(test_stars).write(path_saving_stars)
            self.df.to_parquet(path_saving_stars)

    def _draw_nb_of_companions(self):
        """
        create new column in the global df with the nb of companion for each star
        based on multiplicity fraction and companion frequency
        """
        ### fit multiplicity fraction
        estimated_multiplicity = SIMU_STARS_MODULE.multiplicity_fraction(self.df[self.mass_col_name])/100
        ### fit nf of companions
        estimated_companions_freq = SIMU_STARS_MODULE.companion_frequency(self.df[self.mass_col_name])
        ### boolean array, saving for each star if they are the primary mass of a multiple syst.
        in_multiple_syst = np.random.uniform(size=len(self.df)) < estimated_multiplicity
        nb_syst_multiple = in_multiple_syst.sum()

        ### creating col to save nb aof actual companions
        self.df["nb_companions"] = np.zeros(len(self.df), dtype=int)
        ### draw random nb of companions following multiplicity and companion freq
        self.df.loc[in_multiple_syst, "nb_companions"] = (np.random.poisson((estimated_companions_freq[in_multiple_syst] / estimated_multiplicity[in_multiple_syst]) - 1, size=nb_syst_multiple) +1 ).astype(int)


    def _find_companions(self, id_bin):
        """
        for a given bin id, draw all q values and find companions stars for primeries among singles stars
        write results in the general df
        """

        df_stars = self.df[self.df["iord"]==id_bin]
        bin_index = df_stars.index
        df_stars = df_stars.reset_index(drop=True)
        ### indices of stars with companions
        indices_to_go_over = df_stars.query("nb_companions > 0").index.to_numpy()
        masses = df_stars[self.mass_col_name].to_numpy()
        #masses = df_stars["mass_star"].to_numpy()
        # array with nb of companions
        companions = df_stars.loc[indices_to_go_over, "nb_companions"].to_numpy()
        ### for q_values
        masses_repeated = np.repeat(masses[indices_to_go_over], companions)

        q_values = np.random.uniform(low=0.05, size=len(masses_repeated))
        m_compa = q_values*masses_repeated
        
        df_stars["category"] = np.where(df_stars['nb_companions']==0, 's', 'p')
        df_stars["num_syst"] = np.zeros(len(df_stars))
        index_into_companions = 0
        
        for i_primary, n_companions in zip(indices_to_go_over, companions):
            self.nb_tot_binary_syst += 1
            # mass of companionS to search in the main table 
            m_compas_rdm = m_compa[index_into_companions : index_into_companions+n_companions]
            # look for those masses in the single stars
            for mass_1compa in m_compas_rdm:
                array_diff_mass = (df_stars[df_stars["category"]=='s'][self.mass_col_name] - mass_1compa)
                
                indice_compa_s = np.argmin(abs(array_diff_mass))
                indice_compa = df_stars[df_stars["category"]=='s'].iloc[indice_compa_s].name
                
                # verification that M_compa > M_primary
                if df_stars.loc[indice_compa, self.mass_col_name] > df_stars.loc[i_primary, self.mass_col_name]:
                    df_stars.loc[i_primary, "low_mass_compa"] += 1
                    df_stars.loc[i_primary, "num_syst"] = self.nb_tot_binary_syst
                else:
                    df_stars.loc[indice_compa, "category"] = 'c'
                    df_stars.loc[i_primary, "num_syst"] = self.nb_tot_binary_syst
                    df_stars.loc[indice_compa, "num_syst"] = self.nb_tot_binary_syst
                    # store the (global!) index of the primary star
                    df_stars.loc[indice_compa, "index_primary"] = i_primary+bin_index[0]
                
            index_into_companions += n_companions

        ### save bin's new param into the general df
        new_col = ["category", "num_syst", "low_mass_compa", "index_primary"]
        self.df.loc[bin_index, new_col] = (df_stars.loc[:, new_col]).set_index(bin_index)
        
    
    def make_multiple_systems(self, 
                              params: dict,
                              path_saving_star_syst=None, 
                              path_saving_singles=None, 
                              path_saving_binary_physical_param=None, 
                              path_saving_multiples=None):
        """
        det fraction of binaries or triple systems as a fct of the primary mass, and associate stars together to make stellar multiple systems...
        """
        if path_saving_star_syst is None:
            path_saving_star_syst = self._get_filename_and_path('allstars', params)
        if path_saving_singles is None:
            path_saving_singles = self._get_filename_and_path('singles', params)
        if path_saving_binary_physical_param is None:
            path_saving_binary_physical_param = self._get_filename_and_path('binaries', params)
        if path_saving_multiples is None:
            path_saving_multiples = self._get_filename_and_path('multiples', params)

        ### check if file exist, and if yes if multiplicity col exist
        if os.path.exists(path_saving_star_syst):
            self.df = pd.read_parquet(path_saving_star_syst)
            if 'nb_companions' in list(self.df.columns):
                ### multiple systems already created
                if self.load_general_df:
                    self.df = pd.read_parquet(path_saving_star_syst)
                    print("StarSample.df loaded")
                else:
                    self.df = None
                    print("StarSample.df already exists but will not be charged into raw memory")
            else:
                ### associate to each star a nb of companions
                self._draw_nb_of_companions()
                ### associate each star to a category (preliminar)
                self.df["category"] = np.where(self.df['nb_companions']==0, 's', 'p')
                ### create new columns: flag for low mass companion and index of primary
                self.df["low_mass_compa"] = 0.0
                self.df["index_primary"] = np.nan
                ### loop over every stellar particle...
                for id_bin in tqdm(self.df["iord"].unique()):
                    self._find_companions(id_bin)
                self.df.to_parquet(path_saving_star_syst)

        else:
            ### associate to each star a nb of companions
            self._draw_nb_of_companions()
            ### associate each star to a category (preliminar)
            self.df["category"] = np.where(self.df['nb_companions']==0, 's', 'p')
            ### create new columns: flag for low mass companion and index of primary
            self.df["low_mass_compa"] = 0.0
            self.df["index_primary"] = np.nan
            ### loop over every stellar particle...
            for id_bin in tqdm(self.df["iord"].unique()):
                self._find_companions(id_bin)
            self.df.to_parquet(path_saving_star_syst)

        if os.path.exists(path_saving_singles):
            if self.load_singles_df:
                self.singles = pd.read_parquet(path_saving_singles)
                print("StarSample.singles loaded")
            else:
                print("StarSample.singles already exists but will not be charged into raw memory")
        else:
            self.singles = self.df.query("category == 's'")
            self.singles.reset_index(drop=True, inplace=True)
            self.singles.to_parquet(path_saving_singles)
            
        if os.path.exists(path_saving_binary_physical_param): #path_saving_binaries):
            if self.load_binaries_df:
                self.binaries = pd.read_parquet(path_saving_binary_physical_param) #path_saving_binaries)
                print("StarSample.binaries loaded")
            else:
                print("StarSample.binaries already exists but will not be charged into raw memory")
        else:
            index_p_binary = self.df.query("category == 'p' & nb_companions == 1").index
            df_p_binary = self.df.loc[index_p_binary]
            index_c_binary = self.df.loc[self.df.index_primary.isin(index_p_binary)].index
            df_c_binary = self.df.loc[index_c_binary]
            self.binaries = df_p_binary.merge(df_c_binary, how='outer').drop(columns="index_primary")
            self._assign_all_physical_param_to_binaries()
            self.binaries.to_parquet(path_saving_binary_physical_param) #path_saving_binaries)
            
        if os.path.exists(path_saving_multiples):
            if self.load_multiples_df:
                self.multiples = pd.read_parquet(path_saving_multiples)
                print("StarSample.multiples loaded")
            else:
                print("StarSample.multiples already exists but will not be charged into raw memory")
        else:
            index_p_multiple = self.df.query("category == 'p' & nb_companions > 1").index
            df_p_multiple= self.df.loc[index_p_multiple]
            index_c_multiple = self.df.loc[self.df.index_primary.isin(index_p_multiple)].index
            df_c_multiple = self.df.loc[index_c_multiple]
            self.multiples = df_p_multiple.merge(df_c_multiple, how='outer').drop(columns="index_primary")
            self.multiples.to_parquet(path_saving_multiples)


    def _stellar_type(self):
        """
        add stellar type column in binary df for PRIMARY stars
        different categories: late/normal/early M and FGK (from Winters+19 pr M-stars)
        """
        m = self.binaries[self.mass_col_name]
        lim_mass_stellar_type = [m<0.15, m<0.3, m<0.6, m<1.6]
        classification = ['lM', 'mM', 'eM', 'FGK']
        stellar_type_primary = np.select(lim_mass_stellar_type, classification, 'o')
        
        self.binaries["stellar_type"] = stellar_type_primary
        self.binaries.loc[self.binaries['category']=='c', 'stellar_type'] = 'c'

    def _assign_separation_a(self):
        """
        draw random separation a for every primary star and add it to a new column
        following a lognormal distribution whose param depend on the star type 
        see Offner+23 table 2 (or fig.2) for early/late M and FGK stars
        see Winters+19 fig.20 for mid-M stars
        """
        ### param from Offner+23 and Winters+19
        # in AU
        mu_fgk, mu_eM, mu_lM, mu_mM = np.log10([40, 25, 4, 10.6])
        sig_fgk, sig_eM, sig_lM, sig_mM = 1.5, 1.3, 0.7, 1.1

        ### save index of the primaries of each stellar type 
        #category == 'p' & inutile pcq compa ont type 'c'
        index_primaries_fgk = self.binaries.query("stellar_type == 'FGK'").index
        index_primaries_eM = self.binaries.query("stellar_type == 'eM'").index
        index_primaries_lM = self.binaries.query("stellar_type == 'lM'").index
        index_primaries_mM = self.binaries.query("stellar_type == 'mM'").index
        
        ### draw random a for all primaries, using != param for each stellar type, TRUNCATED to avoid non physical values...
        ### bornes inf and sup for a (in natural unit)
        binf_a = np.log10((2*cst.R_sun.to(u.au)).value)
        bsup_a = np.log10(10**4) # AU
        # limit inf and sup in standard deviation units
        # a and b have to be in unit of standard deviation!
        a_fgk_trunc = 10**st.truncnorm.rvs(a=((binf_a-mu_fgk)/sig_fgk)*np.ones(len(index_primaries_fgk)), 
                                       b=((bsup_a-mu_fgk)/sig_fgk)*np.ones(len(index_primaries_fgk)), 
                                       loc=mu_fgk*np.ones(len(index_primaries_fgk)),
                                       scale=sig_fgk*np.ones(len(index_primaries_fgk)))
        a_eM_trunc = 10**st.truncnorm.rvs(a=((binf_a-mu_eM)/sig_eM)*np.ones(len(index_primaries_eM)), 
                                       b=((bsup_a-mu_eM)/sig_eM)*np.ones(len(index_primaries_eM)), 
                                       loc=mu_eM*np.ones(len(index_primaries_eM)),
                                       scale=sig_eM*np.ones(len(index_primaries_eM)))
        a_lM_trunc = 10**st.truncnorm.rvs(a=((binf_a-mu_lM)/sig_lM)*np.ones(len(index_primaries_lM)), 
                                       b=((bsup_a-mu_lM)/sig_lM)*np.ones(len(index_primaries_lM)), 
                                       loc=mu_lM*np.ones(len(index_primaries_lM)),
                                       scale=sig_lM*np.ones(len(index_primaries_lM)))
        a_mM_trunc = 10**st.truncnorm.rvs(a=((binf_a-mu_mM)/sig_mM)*np.ones(len(index_primaries_mM)), 
                                       b=((bsup_a-mu_mM)/sig_mM)*np.ones(len(index_primaries_mM)), 
                                       loc=mu_mM*np.ones(len(index_primaries_mM)),
                                       scale=sig_mM*np.ones(len(index_primaries_mM)))

        ### assign corresponding indexes
        a_fgk = pd.Series(a_fgk_trunc, index=index_primaries_fgk)
        a_eM = pd.Series(a_eM_trunc, index=index_primaries_eM)
        a_lM = pd.Series(a_lM_trunc, index=index_primaries_lM)
        a_mM = pd.Series(a_mM_trunc, index=index_primaries_mM)

        ### save them into the binary general df
        self.binaries["separation_a"] = np.nan
        self.binaries.loc[index_primaries_fgk, "separation_a"] = a_fgk
        self.binaries.loc[index_primaries_eM, "separation_a"] = a_eM
        self.binaries.loc[index_primaries_lM, "separation_a"] = a_lM
        self.binaries.loc[index_primaries_mM, "separation_a"] = a_mM
        
    def _calculate_period_binaries(self):
        """
        calculate period for all binary systems (self.binaries): save it in the primary star line
        for primaries with a low-mass companion, just consider its mass == 0
        P: Earth year (calculated in sec but converted to be easier)
        a: doit etre en m (conversion from AU) 
        M: doit etre en kg (conversion from M_sun)
        """
        index_primary = self.binaries.query("category == 'p' & low_mass_compa == 0").index
        ### récupérer masses des companions, dans le même ordre que primaries!
        # normalement sort_values garde les index originaux donc devrait fonctionner... (?)
        sorted_p = self.binaries.query("category == 'p' & low_mass_compa == 0").sort_values(by=['num_syst'])
        sorted_c = self.binaries.query("category == 'c'").sort_values(by=['num_syst'])
        #print("nb primaries == nb companions:", len(sorted_p)==len(sorted_c))
        # pas la meme taille pcq exist low mass companions.....

        M = (sorted_p[self.mass_col_name] + sorted_c[self.mass_col_name].values)*(u.M_sun.to(u.kg))
        # period will be in second
        periods = (sorted_p.separation_a*(u.au.to(u.m)))**(3./2.) * (4*np.pi**2 / (cst.G.value * M))**(1./2.)

        self.binaries["period_yr"] = np.nan
        self.binaries.loc[index_primary, "period_yr"] = periods * u.s.to(u.yr)

        ### calculate period for binary systems with a low mass companion
        p_low_mass_compa = self.binaries.query("category == 'p' & low_mass_compa == 1")
        M_lmc = p_low_mass_compa[self.mass_col_name]*(u.M_sun.to(u.kg))
        periods_lmc = (p_low_mass_compa.separation_a*(u.au.to(u.m)))**(3./2.) * (4*np.pi**2 / (cst.G.value * M_lmc))**(1./2.)
        self.binaries.loc[p_low_mass_compa.index, "period_yr"] = periods_lmc * u.s.to(u.yr)


    def _e_max(self, P):
        """
        P: array!
        e_max from eq.3 Moe & Di Stefano 17
        input P in years, convert in days
        ### for P<=2 days e_max=0
        """
        Pdays = P*u.yr.to(u.day)
        emax = 1 - (Pdays/2)**(-2./3.)
        # where e_max < 0 (P < 2 days) change it to 0
        emax[emax<0] = 0
        return emax

    def e_max_indiv(self, P, unit_yr=True):
        """
        P: unique value!!
        e_max from eq.3 Moe & Di Stefano 17
        input P in years, convert in days
        ### not for P<=2 days!
        """
        if unit_yr:
            Pdays = P*u.yr.to(u.day)
        else:
            Pdays = P
        emax = 1 - (Pdays/2)**(-2./3.)
        return emax

    def _assign_excentricity_uniform(self):
        """
        OLD
        draw random excentricity for all binary systems.
        first approx: uniform between 0 and e_max (eq.3 Moe & Di Stefano 17)
        """
        self.binaries["excentricity"] = np.nan
        index_primary = self.binaries.query("category == 'p'").index

        self.test = self.binaries.query("category == 'p'").period_yr.values
        ### draw e_max following uniform distribution:
        e = np.random.uniform(0, self._e_max(self.binaries.query("category == 'p'").period_yr.values))
        self.test = e
        self.binaries.loc[index_primary, 'excentricity'] = pd.Series(data=e, index=index_primary)
        self.binaries.loc[self.binaries.excentricity < 0, 'excentricity'] = 0

    def _assign_excentricity_gaussian(self):
        """
        draw random excentricity for all binary systems.
        first approx: gaussian (same for all P) between 0 and e_max (eq.3 Moe & Di Stefano 17)
        lim P = 2 days : P ~= 0.00548 yrs
        """
        self.binaries["excentricity"] = np.nan
        index_primary = self.binaries.query("category == 'p'").index
        index_primary2 = self.binaries.query("category == 'p' & period_yr >= 0.00548").index
        index_primary2inf = self.binaries.query("category == 'p' & period_yr < 0.00548").index

        ### draw e_max following gaussian distrib (for now same param for all P)
        mu = np.ones(len(index_primary2))*0.5
        sig = np.ones(len(index_primary2))*0.3
        borne_min = np.zeros(len(index_primary2))
        borne_max = self._e_max(self.binaries.query("category == 'p' & period_yr >= 0.00548").period_yr.values)
        self.test = (borne_min-mu)/sig#self._e_max(self.binaries.query("category == 'p'").period_yr.values)
        self.test1 = (borne_max-mu)/sig
        
        e = st.truncnorm.rvs(a=(borne_min-mu)/sig, b=(borne_max-mu)/sig, loc=mu, scale=sig)
                            
        self.test = e
        self.binaries.loc[index_primary2, 'excentricity'] = pd.Series(data=e, index=index_primary2)
        self.binaries.loc[index_primary2inf, 'excentricity'] = 0

    
    def _approx_e_segmentcentral(self, P):
        """
        P: array in years
        return mu(P) for gaussian distrib for 10<=P<1e3 days
        in output P is in DAYS
        """
        return np.log10(P*u.yr.to(u.day))/4. - 0.25

    def _sig_2_a_10_jours(self, P):
        """
        P: array in years
        return sigma(P) for gaussian distrib for 2<=P<10 days
        linear in log from 0 to sigma_segmentcentral (self.sigma_gaussian_e)
        in output P in DAYS
        """
        return (np.log10(P*u.yr.to(u.day)) - 0.3) * self.sigma_gaussian_e/0.7

    def _assign_excentricity_segment(self):
        """
        draw random excentricity for all binary systems: between 0 and e_max (eq.3 Moe & DS 17)
        try by segment: gaussian mu(P)=0 for 2<P<10 days
                        gaussian mu(logP)= logP/4-0.25 for 10<P<1000 days
                        uniform for P>1000 days
        lim P = 2 days : P ~= 0.00548 yrs
        """
        self.binaries["excentricity"] = np.nan
        # P < 2 days <=> 0.00548 yr
        #index_primary_inf2 = self.binaries.query("category == 'p' & period_yr < 0.00548").index
        primary_inf2 = self.binaries.query("category == 'p' & period_yr < 0.00548")
        # 2 < P < 10 days <=> 0.02738 yr
        #index_primary_inf10 = self.binaries.query("category == 'p' & period_yr >= 0.00548 & period_yr < 0.02738").index
        primary_inf10 = self.binaries.query("category == 'p' & period_yr >= 0.00548 & period_yr < 0.02738")
        # 10 < P < 1000 days <=> 0.02738 yr
        #index_primary_inf1000 = self.binaries.query("category == 'p' & period_yr >= 0.02738 & period_yr < 2.73785").index
        primary_inf1000 = self.binaries.query("category == 'p' & period_yr >= 0.02738 & period_yr < 2.73785")
        # 1000 < P 
        #index_primary_sup1000 = self.binaries.query("category == 'p' & period_yr >= 2.73785").index
        primary_sup1000 = self.binaries.query("category == 'p' & period_yr >= 2.73785")
        
        ##### draw e
        ### P < 2 days: all = 0
        self.binaries.loc[primary_inf2.index, 'excentricity'] = 0
        
        ### 2 < P < 10 days: gaussian centered in 0, std = ?
        mu_inf10 = np.zeros(len(primary_inf10))
        sig_inf10 = self._sig_2_a_10_jours(primary_inf10.period_yr.values) #np.ones(len(primary_inf10))*0.1
        binf_inf10 = np.zeros(len(primary_inf10))
        bsup_inf10 = self._e_max(primary_inf10.period_yr.values)
        e_inf10 = st.truncnorm.rvs(a=(binf_inf10-mu_inf10)/sig_inf10, b=(bsup_inf10-mu_inf10)/sig_inf10, loc=mu_inf10, scale=sig_inf10)
        self.binaries.loc[primary_inf10.index, 'excentricity'] = pd.Series(data=e_inf10, index=primary_inf10.index)
        
        ### 10 < P < 1000 days: gaussian centered in mu(logP), std = ?
        self.test = self._approx_e_segmentcentral(primary_inf1000.period_yr.values)
        mu_inf1000 = self._approx_e_segmentcentral(primary_inf1000.period_yr.values)
        sig_inf1000 = np.ones(len(primary_inf1000))*self.sigma_gaussian_e
        binf_inf1000 = np.zeros(len(primary_inf1000))
        bsup_inf1000 = self._e_max(primary_inf1000.period_yr.values)
        e_inf1000 = st.truncnorm.rvs(a=(binf_inf1000-mu_inf1000)/sig_inf1000, b=(bsup_inf1000-mu_inf1000)/sig_inf1000, loc=mu_inf1000, scale=sig_inf1000)
        self.binaries.loc[primary_inf1000.index, 'excentricity'] = pd.Series(data=e_inf1000, index=primary_inf1000.index)
        
        ### 1000 days < P: uniform
        e_sup1000 = st.uniform.rvs(loc=0., scale=self._e_max(primary_sup1000.period_yr.values))#, size=len(primary_sup1000))
        self.binaries.loc[primary_sup1000.index, 'excentricity'] = pd.Series(data=e_sup1000, index=primary_sup1000.index)
        

    def _assign_all_physical_param_to_binaries(self):
        """
        (only in BINARY df!)
        if file doesn't already exist: 
            - determine the stellar type of all primary stars
            - draw separation a
            - compute associated period
            - draw eccentricity
        """
        
        ### stellar type because distrib de a en %
        self._stellar_type()
        
        ### draw random separation a for every primary star
        self._assign_separation_a()
        self._calculate_period_binaries()

        ### draw random eccentricity following 4 regimes-distribution
        # (possible to specify sigma_e for 3rd segment with self.sigma_gaussian_e, default=0.3)
        self._assign_excentricity_segment()  



#####################################################
#####################################################
#####################################################
############## PlanetSample sub-class ###############
#####################################################
#####################################################
#####################################################

class PlanetSample(StarSample):
    """
    Extension of the StarSample class, with planet-related methods
    """

    def __init__(self, *args, **kwargs): #,
            #file_name, file_fecha, extra_info_file,
            #df, singles, binaries, multiples,
            #feh_col_name, teff_col_name, logg_col_name, radius_star_col_name, mass_col_name):
        super().__init__(*args, **kwargs)
                        #file_name, file_fecha, extra_info_file, df, singles, binaries, multiples,
                        #feh_col_name, teff_col_name, logg_col_name, radius_star_col_name, mass_col_name)
        self.planets = None
        self.planetary_systems = None

    def _get_filename_and_path(self, 
                               which_table: str, 
                               params: dict) -> str:
        """
        return path + filename for a given table 
        to load or save it

        available tables:
            - [df] ([file identificator]): [str to input in 'which_table']
            - self.singles ('_SINGLES_'): singles
            - self.planets ('_PLANETS_'): planets
            - self.binaries ('_BINARIES_'): binaries
            - self.multiples ('_MULTIPLES_'): multiples
            - self.df ('_ALLSTARS_'): allstars (only from simulation)
        """
        str_to_filecode = {'singles': '_SINGLES_',
                           'singlesP': '_SINGLES_and_PLANETSstats_',
                           'planets': '_PLANETS_',
                           'planetary_syst': '_PLANETARY_SYST_',
                           'binaries': '_BINARIES_',
                           'multiples': '_MULTIPLES_',
                           'allstars': '_ALLSTARS_'}
        
        filename = params['OUTPUT']['sample_name'] + str_to_filecode[which_table] \
            + params['OUTPUT']['date'] + '_' + params['OUTPUT']['extra_info_in_filename'] + '.parquet'
        
        return params['OUTPUT']['path'] + filename



    def _draw_nb_of_planets_singles(self, 
                                    params: dict):
        """
        only for single stars
        given the fraction of stars having the != types of planets, depending on their MASS (table 3 form Burn+21) and their METALLICITY (Narang+18)
        assign them a certain nb of planet of each type (table 4 Burn+21)
        minimum and maximum threshold (-1.0 and 0.6)
        """
        #### use occ rate and multiplicity models declared in .params file
        [occ_rate_Earthlike_MFeh, occ_rate_SEnept_MFeh, occ_rate_giants_MFeh] = OCC_RATE_MODULE.choice_occrate_fct(params['INPUT_PLANET_CREATION']['custom_occ_rate'])
        [multiplicity_Earthlike, multiplicity_SE_nept, multiplicity_giants] = OCC_RATE_MODULE.choice_multiplicity_fct(params['INPUT_PLANET_CREATION']['custom_multiplicity'])

        ### create new columns with flag for presence of exoplanets of != types
        self.singles["have_Earthlike"] = occ_rate_Earthlike_MFeh(self.singles[self.feh_col_name], self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))
        #fraction_Earthlike(self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))
        self.singles["have_SE_nept"] = occ_rate_SEnept_MFeh(self.singles[self.feh_col_name], self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))
        #fraction_SE_nept(self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))
        self.singles["have_giants"] = occ_rate_giants_MFeh(self.singles[self.feh_col_name], self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))
        #fraction_giants(self.singles[self.mass_col_name]) > np.random.uniform(size=len(self.singles))

        ### initialize empty (zeros) columns to store the number of planet of each type
        self.singles["nb_Earthlike"] = np.zeros(len(self.singles))
        self.singles["nb_SE_nept"] = np.zeros(len(self.singles))
        self.singles["nb_giants"] = np.zeros(len(self.singles))

        ### if presence==True, draw nb of planets of the corresponding type (Poisson)
        self.singles.loc[self.singles.have_Earthlike, "nb_Earthlike"] = np.random.poisson(multiplicity_Earthlike(self.singles.loc[self.singles.have_Earthlike, self.mass_col_name].values))
        self.singles.loc[self.singles.have_SE_nept, "nb_SE_nept"] = np.random.poisson(multiplicity_SE_nept(self.singles.loc[self.singles.have_SE_nept, self.mass_col_name].values))
        self.singles.loc[self.singles.have_giants, "nb_giants"] = np.random.poisson(multiplicity_giants(self.singles.loc[self.singles.have_giants, self.mass_col_name].values))


    def _assign_planetary_type(self):
        """
        df_1syst: lines corresponding to all the planets belonging to the same hoststar
        hoststar: pd.Series from sample_stars.singles, line with hoststar param
        """
        ### create an empty list (actual list and no numpy array bc np.append() is not efficient enough...)
        planet_types = []
        ### try to use only lists to make the code faster (?)
        liste_nbEarth = self.singles.loc[:, 'nb_Earthlike'].values.astype(int)
        liste_nbSE = self.singles.loc[:, 'nb_SE_nept'].values.astype(int)
        liste_nbgiants = self.singles.loc[:, 'nb_giants'].values.astype(int)
        ### for each hoststar, add every planets to the list (6*"Earthlike", 3*"SE_nept", etc)
        for hs in tqdm(range(len(self.singles))):
            planet_types.append(["giant"]*liste_nbgiants[hs]
                                     + ["SE_nept"]*liste_nbSE[hs]
                                     + ["Earthlike"]*liste_nbEarth[hs])
            ############### test inverser earths and SE
            #planet_types.append(["SE_nept"]*liste_nbSE[hs]
            #                         + ["Earthlike"]*liste_nbEarth[hs]
            #                         + ["giant"]*liste_nbgiants[hs])
        ### flatten the list:
        flat_planet_types = [x for xs in planet_types for x in xs]
        ### save the list into the df_planets
        self.planets["planet_type"] = flat_planet_types

    def _draw_initial_params(self, 
                            params: dict):
        """
        draw M (in Mearth) and period (in days) for each planet, using appropriate gaussian depending on the planetary type
        save them in the df
        TO DO:
        - assign eccentricity: which distribution??
        """

        means_logP_logM, covs_logP_logM = EXOPL_MODULE.get_gaussians_params(params['INPUT_PLANET_CREATION']['mass_period_distrib'])
        ##### draw M-P point:
        ### Earth-like: gaussian 3
        EL = self.planets[self.planets['planet_type'] == 'Earthlike']
        MP_EL = np.random.multivariate_normal(means_logP_logM[3], covs_logP_logM[3], size=len(EL))
        self.planets.loc[EL.index, "mass_planet"] = 10**MP_EL[:,1]
        self.planets.loc[EL.index, "period_planet"] = 10**MP_EL[:,0]
        ### SE-nept: gaussian 2
        SE = self.planets[self.planets['planet_type'] == 'SE_nept']
        MP_SE = np.random.multivariate_normal(means_logP_logM[2], covs_logP_logM[2], size=len(SE))
        ### giants: gaussians 0 and 1 --> COMBINE THEM....
        self.planets.loc[SE.index, "mass_planet"] = 10**MP_SE[:,1]
        self.planets.loc[SE.index, "period_planet"] = 10**MP_SE[:,0]
        ######## Hot or Cold Jupiter?
        HCG = self.planets[self.planets['planet_type'] == 'giant']
        gaussian_giant = np.random.uniform(size=len(HCG))
        g = pd.Series(gaussian_giant, index=HCG.index) # 1: WJ/CJ; 0:HJ
        ### save them into two Series: one with HJ and one with CJ/WJ WITH INDEXES!
        g_HJ = g[g<=0.1]
        g_CJ = g[g>0.1]
        MP_g_HJ = np.random.multivariate_normal(means_logP_logM[0], covs_logP_logM[0], size=len(g_HJ))
        MP_g_CJ = np.random.multivariate_normal(means_logP_logM[1], covs_logP_logM[1], size=len(g_CJ))
        
        self.planets.loc[g_HJ.index, "mass_planet"] = 10**MP_g_HJ[:,1]
        self.planets.loc[g_HJ.index, "period_planet"] = 10**MP_g_HJ[:,0]
        self.planets.loc[g_CJ.index, "mass_planet"] = 10**MP_g_CJ[:,1]
        self.planets.loc[g_CJ.index, "period_planet"] = 10**MP_g_CJ[:,0]

        # compute semi mayor axis
        self.planets["a_AU"] = self._calcul_semi_mayor_axis(self.planets)
        

        ### ADD HERE LATER: draw eccentricity...

    def _redraw_illegal_params(self, 
                               illegal_pl,
                               params: dict):
        """
        draw M (in Mearth) and period (in days) for planet with illegal params, using appropriate gaussian depending on the planetary type
        """
        means_logP_logM, covs_logP_logM = EXOPL_MODULE.get_gaussians_params(params['INPUT_PLANET_CREATION']['mass_period_distrib'])
        ##### draw M-P point:
        ### Earth-like: gaussian 3
        EL = illegal_pl[illegal_pl["planet_type"] == 'Earthlike']
        MP_EL = np.random.multivariate_normal(means_logP_logM[3], covs_logP_logM[3], size=len(EL))
        self.planets.loc[EL.index, "mass_planet"] = 10**MP_EL[:,1]
        self.planets.loc[EL.index, "period_planet"] = 10**MP_EL[:,0]
        ### SE-nept: gaussian 2
        SE = illegal_pl[illegal_pl['planet_type'] == 'SE_nept']
        MP_SE = np.random.multivariate_normal(means_logP_logM[2], covs_logP_logM[2], size=len(SE))
        ### giants: gaussians 0 and 1 --> COMBINE THEM....
        self.planets.loc[SE.index, "mass_planet"] = 10**MP_SE[:,1]
        self.planets.loc[SE.index, "period_planet"] = 10**MP_SE[:,0]
        ######## Hot or Cold Jupiter?
        HCG = illegal_pl[illegal_pl['planet_type'] == 'giant']
        gaussian_giant = np.random.uniform(size=len(HCG))
        g = pd.Series(gaussian_giant, index=HCG.index) # 1: WJ/CJ; 0:HJ
        ### save them into two Series: one with HJ and one with CJ/WJ WITH INDEXES!
        g_HJ = g[g<=0.1]
        g_CJ = g[g>0.1]
        MP_g_HJ = np.random.multivariate_normal(means_logP_logM[0], covs_logP_logM[0], size=len(g_HJ))
        MP_g_CJ = np.random.multivariate_normal(means_logP_logM[1], covs_logP_logM[1], size=len(g_CJ))
        
        self.planets.loc[g_HJ.index, "mass_planet"] = 10**MP_g_HJ[:,1]
        self.planets.loc[g_HJ.index, "period_planet"] = 10**MP_g_HJ[:,0]
        self.planets.loc[g_CJ.index, "mass_planet"] = 10**MP_g_CJ[:,1]
        self.planets.loc[g_CJ.index, "period_planet"] = 10**MP_g_CJ[:,0]

        ################# recalculer semi-mayor axis... comment marchait avant alors??
        # on recalcule tout d'un coup flemme de trier pr l'instant..
        self.planets.loc[:, "a_AU"] = self._calcul_semi_mayor_axis(self.planets)

    
    def _redraw_illegal_planets(self,
                                params: dict):
        """
        redraw M-P param for illegal planets, being:
        - planets with M > 1e4 Mearth
        - planets with semi mayor axis < 2*Rstar
        """
        ### det which planets have to be redrawn
        illegal_planets = self.planets[(self.planets['mass_planet'] >= 1e4) | (self.planets['a_AU']*u.au.to(u.Rsun) <= 2*self.planets['radius_star'])]
        print("     nb of illegal planets:", len(illegal_planets), ", start redraw...")
        ### redraw param for illegal ones, until ok
        nb_it = 0
        while (len(illegal_planets) > 0) and (nb_it < 100):
            nb_it += 1
            # redraw params and save them into general df
            self._redraw_illegal_params(illegal_planets, params)
            # check illegality
            illegal_planets = self.planets[(self.planets['mass_planet'] >= 1e4) | (self.planets['a_AU']*u.au.to(u.Rsun) <= 2*self.planets['radius_star'])]
            # repeat if necessary..
        print(f"     redraw finished! ({str(nb_it)} redraws: {str(len(illegal_planets))} remaining illegal exoplanets (descarted))")
        ### set all planet params to NaN if still illegals
        self.planets.loc[illegal_planets.index, ['mass_planet', 'period_planet', 'a_AU']] = np.nan

    def _calculate_semi_mayor_axis(self):
        """
        deprecated?
        using Kepler 3rd law
        P in seconds (converted from years)
        M in kg (converted from Msun)
        returns a in AU (calculated in m)
        """
        ############################## mass etoile ptn pas planete......
        a_m = (cst.G.value * (self.planets.mass_planet*u.Mearth.to(u.kg) + self.planets.mass_star*u.Msun.to(u.kg)) / (4*np.pi**2))**(1./3.) * (self.planets.period_planet*u.day.to(u.s))**(2./3.)
        return a_m*u.m.to(u.AU)

    def _calcul_semi_mayor_axis(self, df_plnts):
        """
        compute a for a given set of planets
        using Kepler 3rd law
        P in seconds (converted from years)
        M in kg (converted from Msun)
        returns a in AU (calculated in m)
        """
        ############################## mass etoile ptn pas planete......
        a_m = (cst.G.value * (df_plnts.mass_planet*u.Mearth.to(u.kg) + df_plnts.mass_star*u.Msun.to(u.kg)) / (4*np.pi**2))**(1./3.) * (df_plnts.period_planet*u.day.to(u.s))**(2./3.)
        return a_m*u.m.to(u.AU)  


    def _calcul_lim_exclusion_zone(self, df_plnts):
        """
        compute EZ for a given set of planets
        from a_AU: a_peri, a_apo (periastre and apoastre)
        then exclusion zone defined as:
            r_min = a_peri-dist(L1) = a_peri*(1-eps)
            r_max = a_apo+dist(L2) = a_apo*(1+eps)
            with eps = (q/3)**(1/3) and q = m1/(m1+m2)
        """
        r_peri = df_plnts.a_AU * (1-df_plnts.eccentricity)
        r_apo = df_plnts.a_AU * (1+df_plnts.eccentricity)
    
        ### mass ratio (UNITS!!)
        q = df_plnts.mass_planet / (df_plnts.mass_planet+df_plnts.mass_star*u.Msun.to(u.Mearth))
        ### epsilon
        eps = (q/3)**(1./3.)
        ### L1/L2
        r_min = r_peri * (1 - eps)# + eps**2/3. + eps**3/9.)
        r_max = r_apo * (1 + eps)# + eps**2/3. - eps**3/9.)
        # en AU!
        return r_min, r_max    
    
    def _assign_physical_param_planets(self,
                                        params: dict):
        """
        create(/load) a new df with one line per planet, keep track of the planetary system/host star id
        TO DO:
        >>> for each planetary system:
            + draw random couple of param M-P from the bigaussian associated to the planetary type
            - AND draw a random eccentricity (following which distrib??)
            + save it in the table
            + calculate the associated distance from the host star and the exclusion zone* 
              *(from periapsis-dist(L1) to apoapsis+dist(L2)
            - repeat for all the other stars in the system 
              BUT check if they don't belong to a previous exclusion zone (if yes, redraw...)
        TO DO after:
        - draw eccentricity
        - calculate the associated exclusion zone
        - check if they are compatible
        """
        print("assign planetary types...")
        ### transfer planetary types in the planet df
        self._assign_planetary_type()

        print("draw initial parameters")
        ### draw M and P following bi-gaussian distributions
        self._draw_initial_params(params)
        
        print("REDRAW illegal planets...")
        ### REDRAW M-P param for illegal planets (for now: M>1e4Mearth or a_pl < 2*Rstar)
        self._redraw_illegal_planets(params)  
             
        ### assign eccentricity (for now all == 0, change later!)
        self.planets["eccentricity"] = np.zeros(len(self.planets))
        
        #print("compute exclusion zone limits...")
        ### compute exclusion zone for each planet...
        #self.planets["dist_min"], self.planets["dist_max"] = self._calcul_lim_exclusion_zone(self.planets)
        

    def _ComputeInsolation(self):
        """
        INPUT:
            Rs: stellar radius [Rsun], must be converted in m
            a_AU: semi major axis of the planet [AU]
            Teff: stellar effective temperature [K]
        OUTPUT:
            lum: stellar luminosity [L_sun]
            insol: insolation received by the planet [I_sun]
        """
        #lum = (Rs**2.)*((Teff/5778)**4.) #which units??
        ## tout en SI
        Rs = self.planets.radius_star*u.Rsun.to(u.m)
        lum = 4*np.pi*(Rs**2.)*(self.planets.teff_star**4.)*cst.sigma_sb.value
        lum = lum / cst.L_sun.value
        #insol = lum*((1.*u.au/a)**2.)
        insol = lum / ((self.planets.a_AU)**2.)
        return lum, insol
    
    def _ComputeInsolation_fromLum(self):
        """
        same as above but when we already have he luminosity value
        INPUT:
            a_AU: semi major axis of the planet [AU]
            lum: stellar luminosity [L_sun]
        OUTPUT:
            insol: insolation received by the planet [I_sun]
        """
        ## tout en SI
        
        lum = self.planets['lum_star'] # in L_sun
        #insol = lum*((1.*u.au/a)**2.)
        insol = lum / ((self.planets.a_AU)**2.)
        return lum, insol


    def _calculateEffectiveFluxBoundary(self):
        """ 
        for plot purposes
        ---------------
        Use to get the two boundary lines of HZ. Obtained from https://github.com/fenrir-lin/exoPlot. 
        Formula from https://www.annualreviews.org/doi/10.1146/annurev-astro-082214-122238 
        """
        # inner limit: recent Venus
        SUN_IN = 1.7665
        A_IN = 1.3351E-4
        B_IN = 3.1515E-9
        C_IN = -3.3488E-12
        # outer limit: early Mars
        SUN_OUT = 0.324
        A_OUT = 5.3221E-5
        B_OUT = 1.4288E-9
        C_OUT = -1.1049E-12

        T = self.planets['teff_star'] - 5780 # Teff and T are in K
        Seff_in = SUN_IN + A_IN*T + B_IN*(T**2) + C_IN*(T**3)
        Seff_out = SUN_OUT + A_OUT*T + B_OUT*(T**2) + C_OUT*(T**3)
        # Seff in unit of S_sun
        return Seff_in, Seff_out


    def _calculate_star_radius(self):
        """
        INPUT:
            star_teff [K]: can be either a single value or a df column
            star_l: can be either a single value or a df column
                    assumed to be in unit of L_sun
                    
        return stellar radius based on Stefan Boltzmann law: L = 4*pi * R^2 * sigma_sb * T_eff^4
        in Rsun units
        """
        ###################################
        
        luminosity = (self.planets['lum_star'])*cst.L_sun.value
        star_teff = self.planets['teff_star']
        return (np.sqrt(luminosity /(4*np.pi*cst.sigma_sb.value))/(star_teff**2))*u.m.to(u.Rsun)

    def _sig_radinf10(self, rad):
        return 0.08 * rad**3.57

    def _sig_radinf138(self, rad):
        return 0.15 * rad**1.49

    def _sig_radsup138(self, rad):
        return 4 * rad**100
    

    def _sig_massinf10(self, masses):
        return 0.08 * masses**0.28

    def _sig_massinf138(self, masses):
        return 0.15 * masses**0.67

    def _sig_masssup138(self, masses):
        return 4 * masses**0.01


    def _Convert_pl_mass_to_radius_realistic(self, 
                                             mass_col='mass_planet', 
                                             radius_col='radius_planet'):
        """
        Convert planet masses (in Mearth) in planet radius (in Rearth),
        based on relation derived by L. Parc+24
        dispersion det by eye
        17/01/25: para los planetas entre los 2 primeros slopes, randomly det à laquelle appartiennent 
        """

        self.planets[radius_col] = np.zeros(len(self.planets))
        index_segment1 = self.planets[self.planets[mass_col] <= 5].index
        index_segment2 = self.planets[(self.planets[mass_col] > 14) & (self.planets[mass_col] <= 138)].index
        # randomly distribute the ones between 7 and 15 Mearth
        in_between = self.planets[(self.planets[mass_col] > 5) & (self.planets[mass_col] <= 14)]
        bottom = in_between.sample(frac=0.5)
        top = in_between.loc[~in_between.index.isin(bottom.index)]
        index_segment1 = index_segment1.append(bottom.index)
        index_segment2 = index_segment2.append(top.index)
        index_segment3 = self.planets[self.planets[mass_col] > 138].index

        ### param mean and std for each segment
        mu_inf10 = 1.02 * self.planets.loc[index_segment1, mass_col]**0.28
        sig_inf10 = self._sig_massinf10(self.planets.loc[index_segment1, mass_col].values) #np.ones(len(primary_inf10))*0.1
        binf_inf10 = np.ones(len(index_segment1))*0.1
        bsup_inf10 = np.ones(len(index_segment1))*3
        R_inf10 = st.truncnorm.rvs(a=np.array((binf_inf10-mu_inf10)/sig_inf10, dtype='float'), b=np.array((bsup_inf10-mu_inf10)/sig_inf10, dtype='float'), 
                                   loc=mu_inf10, scale=sig_inf10)

        mu_inf138 = 0.61 * self.planets.loc[index_segment2, mass_col]**0.67
        sig_inf138 = self._sig_massinf138(self.planets.loc[index_segment2, mass_col].values) #np.ones(len(primary_inf10))*0.1
        binf_inf138 = np.ones(len(index_segment2))*2
        bsup_inf138 = np.ones(len(index_segment2))*20
        R_inf138 = st.truncnorm.rvs(a=np.array((binf_inf138-mu_inf138)/sig_inf138, dtype='float'), b=np.array((bsup_inf138-mu_inf138)/sig_inf138, dtype='float'), 
                                    loc=mu_inf138, scale=sig_inf138)

        mu_sup138 = 11.9 * self.planets.loc[index_segment3, mass_col]**0.01
        sig_sup138 = self._sig_masssup138(self.planets.loc[index_segment3, mass_col].values) #np.ones(len(primary_inf10))*0.1
        binf_sup138 = np.ones(len(index_segment3))*8
        bsup_sup138 = np.ones(len(index_segment3))*25
        R_sup138 = st.truncnorm.rvs(a=np.array((binf_sup138-mu_sup138)/sig_sup138, dtype='float'), b=np.array((bsup_sup138-mu_sup138)/sig_sup138, dtype='float'), 
                                    loc=mu_sup138, scale=sig_sup138)

        self.planets.loc[index_segment1, radius_col] = R_inf10.astype('float64')
        self.planets.loc[index_segment2, radius_col] = R_inf138.astype('float64')
        self.planets.loc[index_segment3, radius_col] = R_sup138.astype('float64')

        return self.planets[radius_col]

    def _Convert_pl_mass_to_radius_simple(self):
        """
        (deprecated)
        Convert planet masses (in Mearth) in planet radius (in Rearth),
        using relation derived by L. Parc+24
        """
        self.planets['radius_planet'] = np.zeros(len(self.planets))

        index_segment1 = self.planets[self.planets['mass_planet'] <= 10].index
        index_segment2 = self.planets[(self.planets['mass_planet'] > 10) & (self.planets['mass_planet'] <= 138)].index
        index_segment3 = self.planets[self.planets['mass_planet'] > 138].index

        self.planets.loc[index_segment1, 'radius_planet'] = 1.02 * self.planets.loc[index_segment1, 'mass_planet']**0.28
        self.planets.loc[index_segment2, 'radius_planet'] = 0.61 * self.planets.loc[index_segment2, 'mass_planet']**0.67
        self.planets.loc[index_segment3, 'radius_planet'] = 11.9 * self.planets.loc[index_segment3, 'mass_planet']**0.01

        print('done')

    def _Convert_pl_radius_to_mass_realistic(self, mass_col='mass_planet', radius_col='radius_planet', simple=False):
        """
        used??
        Convert planet radii (in Rearth) in planet mass (in Mearth),
        based on relation derived by L. Parc+24
        dispersion det by eye
        17/01/25: para los planetas entre los 2 primeros slopes, randomly det à laquelle appartiennent 
        """

        self.planets[mass_col] = np.zeros(len(self.planets))
        index_segment1 = self.planets[self.planets[radius_col] <= 1.9].index
        index_segment2 = self.planets[(self.planets[radius_col] > 2.1) & (self.planets[radius_col] <= 13)].index
        # randomly distribute the ones between 7 and 15 Mearth
        in_between = self.planets[(self.planets[radius_col] > 1.9) & (self.planets[radius_col] <= 2.1)]
        bottom = in_between.sample(frac=0.5)
        top = in_between.loc[~in_between.index.isin(bottom.index)]
        index_segment1 = index_segment1.append(bottom.index)
        index_segment2 = index_segment2.append(top.index)
        index_segment3 = self.planets[self.planets[radius_col] > 13].index

        ### param mean and std for each segment
        mu_inf10 = 0.98 * self.planets.loc[index_segment1, radius_col]**3.57
        sig_inf10 = self._sig_radinf10(self.planets.loc[index_segment1, radius_col].values) #np.ones(len(primary_inf10))*0.1
        binf_inf10 = np.ones(len(index_segment1))*0.1
        bsup_inf10 = np.ones(len(index_segment1))*14

        #print((binf_inf10-mu_inf10)/sig_inf10)
        if simple:
            R_inf10 = mu_inf10
        else:
            R_inf10 = st.truncnorm.rvs(a=np.array((binf_inf10-mu_inf10)/sig_inf10, dtype='float'), b=np.array((bsup_inf10-mu_inf10)/sig_inf10, dtype='float'), 
                                    loc=mu_inf10, scale=sig_inf10)

        mu_inf138 = 1.64 * self.planets.loc[index_segment2, radius_col]**1.49
        sig_inf138 = self._sig_radinf138(self.planets.loc[index_segment2, radius_col].values) #np.ones(len(primary_inf10))*0.1
        binf_inf138 = np.ones(len(index_segment2))*5
        bsup_inf138 = np.ones(len(index_segment2))*138
        if simple:
            R_inf138 = mu_inf138
        else:
            R_inf138 = st.truncnorm.rvs(a=np.array((binf_inf138-mu_inf138)/sig_inf138, dtype='float'), b=np.array((bsup_inf138-mu_inf138)/sig_inf138, dtype='float'), 
                                    loc=mu_inf138, scale=sig_inf138)

        mu_sup138 = 10**(-45) * self.planets.loc[index_segment3, radius_col]**50
        sig_sup138 = self._sig_radsup138(self.planets.loc[index_segment3, radius_col].values) #np.ones(len(primary_inf10))*0.1
        binf_sup138 = np.ones(len(index_segment3))*138
        bsup_sup138 = np.ones(len(index_segment3))*10000
        if simple:
            R_sup138 = mu_sup138
            #print(R_sup138)
            #return mu_sup138
            R_sup138.loc[mu_sup138[mu_sup138>1000].index.values] = 10**np.random.uniform(2, 3, len(mu_sup138[mu_sup138>1000].index.values))
            #print(R_sup138)

        else:
            R_sup138 = st.truncnorm.rvs(a=np.array((binf_sup138-mu_sup138)/sig_sup138, dtype='float'), b=np.array((bsup_sup138-mu_sup138)/sig_sup138, dtype='float'), 
                                    loc=mu_sup138, scale=sig_sup138)

        self.planets.loc[index_segment1, mass_col] = R_inf10
        self.planets.loc[index_segment2, mass_col] = R_inf138
        self.planets.loc[index_segment3, mass_col] = R_sup138

        return self.planets[mass_col]

    #################### create planets around input stars
    def create_planets_custom(self, 
                              params: dict):
        """
        init_params: dictionary with input parameters, if not in init_params init file
            should have: sample name, 

        23/09/25: add params argument
        _assign_planet_type takes approx 3 seconds 
        rest is 10~20 seconds, try to improve it...?
        draw_initial_param is slow...

        25/11/24: add boolean save_pl, default=True (to no alter previous utilisations of this fct)
                  if False, don't save to disk the planet file (eavy), but do save the singles+pl file
        """
        
        path_saving_planets = self._get_filename_and_path('planets', params)
        path_saving_singles_wth_planetsstats = self._get_filename_and_path('singlesP', params)

        print(">>> Checking file existence...")
        
        ### if file already exists just load it into the planets df
        if os.path.exists(path_saving_planets):
            print(f">>> Files {path_saving_planets} and {path_saving_singles_wth_planetsstats} already exist, loaded to self.singles and self.planets")
            print("IF YOU WANT TO RERUN IT ANYWAY: delete the files or add a 'extra_info_in_filename' in the .params file")
            self.planets = pd.read_parquet(path_saving_planets)
            self.singles = pd.read_parquet(path_saving_singles_wth_planetsstats)
        else:
            if os.path.exists(path_saving_singles_wth_planetsstats):
                print(">>> star file already exists, loaded to self.singles")
                self.singles = pd.read_parquet(path_saving_singles_wth_planetsstats)
                if 'nb_Earthlike' not in self.singles.columns:
                    print("> assign each single star a number of planets of each type and save")
                    ### draw proba and multiplicity of the != planetary types (for singles stars only)
                    self._draw_nb_of_planets_singles(params)
                    self.singles.to_parquet(path_saving_singles_wth_planetsstats)
            else:
                print("> table with planets and single stars will be saved in:")
                print("%s and %s"%(path_saving_planets, path_saving_singles_wth_planetsstats))
                print('-------')
                print(">>> Creating new file with all input stars + nb of planets")
                print("> assign each single star a number of planets of each type and save")
                ### draw proba and multiplicity of the != planetary types (for singles stars only)
                self._draw_nb_of_planets_singles(params)
                self.singles.to_parquet(path_saving_singles_wth_planetsstats)
                print("single stars file saved to %s"%(path_saving_singles_wth_planetsstats))

            print('-------')
            print(">>> Creating the exoplanets table")
            ### create planets df
            self.planets = pd.DataFrame(columns=['id_hoststar', 'mass_star', 'radius_star', 'teff_star', 'feh', 'planet_type', 'mass_planet', 'period_planet', 'eccentricity', 'a_AU'])
            self.planets[['id_hoststar', 'mass_star']] = self.singles[self.mass_col_name].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()
            self.planets[['id_hoststar', 'feh']] = self.singles[self.feh_col_name].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()

            ### add other stellar param: teff, lum, to compute radius
            if self.teff_col_name != '':
                self.planets[['id_hoststar', 'teff_star']] = self.singles[self.teff_col_name].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()
            if self.lum_col_name != '':
                self.planets[['id_hoststar', 'lum_star']] = self.singles[self.lum_col_name].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()
            
            if self.radius_star_col_name != '':
                self.planets[['id_hoststar', 'radius_star']] = self.singles[self.radius_star_col_name].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()
            else:
                if (self.teff_col_name != '') and (self.lum_col_name != ''):
                    print("> estimating stellar radius (with T_eff and luminosity)")
                    ### compute radius
                    self.planets["radius_star"] = self._calculate_star_radius()
                    self.radius_star_col_name = "radius_star"
                else:
                    print("not able to access/compute the stellar radii, some quantities will not be calculated...")
            ### add any extra stellar parameters in the 'extra_stellar_parameters' list from 'init_custom.py' file     
            if len(self.extra_stellar_params) > 0:
                for p in self.extra_stellar_params:
                    self.planets[['id_hoststar', p]] = self.singles[p].repeat((self.singles.nb_Earthlike + self.singles.nb_SE_nept + self.singles.nb_giants).values.astype(int)).reset_index()

            ### assign physical param: type, mass, period, eccentricity
            print(">> assigning physical parameters to all planets (mass, period, (eccentricity), etc)")
            self._assign_physical_param_planets(params)

            ### add HZ related columns...
            if self.radius_star_col_name != '':
                print("> Calculating insolation received by the planets, and if they fall into the CHZ")
                ### compute planet radius...
                if 'radius_planet' not in self.planets.columns:
                    print("Conversion of planet mass to radius (may take a while...)")
                    t_conv_M_R = timer()
                    self._Convert_pl_mass_to_radius_realistic()
                    print("took %s sec"%(timer()-t_conv_M_R))
                
                print("insolation and HZ...")
                # compute insolation received by the exoplanet(s) from their host star
                self.planets['insolation'] = self._ComputeInsolation()[1]
                if self.lum_col_name != '':
                    self.planets['insolation_fromLum'] = self._ComputeInsolation_fromLum()[1]            
                ### compute min and max insolation associated with HZ
                self.planets['Seff_in'], self.planets['Seff_out'] = self._calculateEffectiveFluxBoundary()
                # empty initialisation
                self.planets['in_HZ'] = False
                # check if belong to HZ
                self.planets.loc[((self.planets['insolation']<=self.planets['Seff_in']) 
                     & (self.planets['insolation']>=self.planets['Seff_out'])), 'in_HZ'] = True                
            else:
                print("> Stellar radius not found: insolation and belonging to the HZ were not computed...")
            
            self.planets.to_parquet(path_saving_planets)
            print("END, file saved to %s"%(path_saving_planets))


    def load_previously_computed_files(self, 
                                       params: dict, 
                                       load_allstars=False, 
                                       load_singles=False, 
                                       load_binaries=False, 
                                       load_multiples=False, 
                                       load_planets=True, 
                                       load_singles_with_planetcols=True):
        """
        load into df previously computed/saved files
        only working for files generated by makefile.py
        INPUT:
            sample_name, fecha: depend on the file we want to load (sample_name refeers to location in the MW, fecha the generation date)
        """
    
        if load_allstars == True:
            self.df = self._get_filename_and_path('allstars', params)
        if load_singles == True:
            self.singles = self._get_filename_and_path('singles', params)
        if load_binaries == True:
            self.binaries = self._get_filename_and_path('binaries', params)
        if load_multiples == True:
            self.multiples = self._get_filename_and_path('multiples', params)
        if load_planets == True:
            self.planets = self._get_filename_and_path('planets', params)
        if load_singles_with_planetcols == True:
            self.singles = self._get_filename_and_path('singlesP', params)

    def load_previously_computed_files_PLOTS(self, 
                                       sample_name,
                                       fecha,
                                       load_allstars=False, 
                                       load_singles=False, 
                                       load_binaries=False, 
                                       load_multiples=False, 
                                       load_planets=True, 
                                       load_singles_with_planetcols=True,
                                       path=None):
        """
        load into df previously computed/saved files
        only working for files generated by makefile.py
        INPUT:
            sample_name, fecha: depend on the file we want to load (sample_name refeers to location in the MW, fecha the generation date)
        """
        if path is None:
            path = 'data/'
        if load_allstars == True:
            self.df = pd.read_parquet(path + sample_name + "_ALLSTARS_" + fecha + ".parquet")
        if load_singles == True:
            self.singles = pd.read_parquet( + sample_name + "_SINGLES_" + fecha + ".parquet")
        if load_binaries == True:
            self.binaries = pd.read_parquet(path + sample_name + "_BINARIES_" + fecha + ".parquet")
        if load_multiples == True:
            self.multiples = pd.read_parquet(path + sample_name + "_MULTIPLES_" + fecha + ".parquet")
        if load_planets == True:
            planet_file_path = path + sample_name + "_PLANETS_" + fecha
            if os.path.exists(planet_file_path+"_plRAD.parquet"):
                self.planets = pd.read_parquet(planet_file_path + "_plRAD.parquet")
            else:
                self.planets = pd.read_parquet(planet_file_path + ".parquet")
        if load_singles_with_planetcols == True:
            self.singles = pd.read_parquet(path + sample_name + "_SINGLES_and_PLANETSstats_" + fecha + ".parquet")



    ###### planetary systems study
    def create_planetary_systems(self, 
                                 params: dict):
        """
        INPUT:
            params: Dict with input parameters in 'INPUT_PLANET_CREATION' category
            #sample_name, fecha: param for file names

            #path_saving_planets: path to existing df with all synthetic planet population
            #path_saving_planetary_systems: path where to save planetary syst df
        GOAL:
        - group planets in their planetary systems (1 line/syst)
        - compute coefficients Cv, Cs (defined by Mishra+23)
        """
        #if sample_name is None:
        #    sample_name = self.file_name
        #if fecha is None:
        #    fecha = self.file_date

        path_saving_planets = self._get_filename_and_path('planets', params)
        path_saving_planetary_systems = self._get_filename_and_path('planetary_syst', params)

        if os.path.exists(path_saving_planetary_systems):
            print("Planetary systems already created, loaded into memory")
            self.planetary_systems = pd.read_parquet(path_saving_planetary_systems)

        else:
            print("### creation of planetary system df...")
            ### load planets df 
            self.planets = pd.read_parquet(path_saving_planets)

            print("# sort planets (may take around a minute...)")
            starttime = timer()
            ### sort planets by planetary systems, and inside planetary systems by distance to their star
            sorted_planets_temp = self.planets.sort_values(by=['id_hoststar', 'a_AU'])
            print("took", timer()-starttime)

            ### group by planetary systems
            pl_syst_temp = sorted_planets_temp.groupby(['id_hoststar'])

            print("calculate Cv and Cs coefficients..")
            ### compute Cv and Cs coefficients defined by Mishra+23
            Cv = pl_syst_temp.mass_planet.std()/pl_syst_temp.mass_planet.mean()
            Cv.rename('Cv_Mpl', inplace=True)

            n = pl_syst_temp.size()
            Cs = (np.log10(pl_syst_temp.mass_planet.prod()/pl_syst_temp.mass_planet.first()) - np.log10(pl_syst_temp.mass_planet.prod()/pl_syst_temp.mass_planet.last())) / (n - 1)
            Cs.rename('Cs_Mpl', inplace=True)

            ### we create a new df for the planetary systems... 
            # we want the columns: 'id_hoststar', 'mass_star', 'radius_star', 'logl_star', 'teff_star', 'mh_star', 'age_star', 'nb_planets', 'Cs_Mpl', 'Cv_Mpl'])
            # copy id_hoststar and star properties..
            #self.planetary_systems = pl_syst_temp[['id_hoststar', 'mass_star', 'radius_star', 'logl_star', 'teff_star','mh_star', 'age_star']].first().copy()
            self.planetary_systems = pl_syst_temp[['id_hoststar', 'mass_star', 'radius_star', 'teff_star']].first().copy()
            
            self.planetary_systems.reset_index(drop=True, inplace=True)
            # add nb of planets
            self.planetary_systems['nb_planets'] = pl_syst_temp['mass_planet'].count().values
            self.planetary_systems['total_pl_mass'] = pl_syst_temp['mass_planet'].sum().values
            self.planetary_systems['Cs_Mpl'] = Cs.values
            self.planetary_systems['Cv_Mpl'] = Cv.values

            # classify planetary systems into the 4 categories of Mishra
            self.planetary_systems["plsyst_type"] = np.where(self.planetary_systems['Cs_Mpl'] < -0.2, 'Antiordered', 
                     np.where(self.planetary_systems['Cs_Mpl'] > 0.2, 'Ordered', 
                              np.where(self.planetary_systems['Cv_Mpl'] <= (np.sqrt(self.planetary_systems['nb_planets']-1)/2), 'Similar', 
                                   'Mixed')))
            print("## DONE...")
            self.planetary_systems.to_parquet(path_saving_planetary_systems)
            print("## ... and saved!")


################################################################################################
######### other fcts ###########################################################################
################################################################################################

def read_star_file(path_starfile):
    if os.path.exists(path_starfile):
        if path_starfile.lower().endswith('.csv'):
            try:
                star_data = pd.read_csv(path_starfile)
            except: 
                sys.exit("csv could not be read...")
        elif path_starfile.lower().endswith('.parquet'):
            try:
                star_data = pd.read_parquet(path_starfile)
            except: 
                sys.exit("parquet could not be read...")
        elif path_starfile.lower().endswith(('.fit', '.fits')):
            try:
                star_data = Table.read(path_starfile).to_pandas()
            except: 
                sys.exit("fits could not be read...")
        else:
            sys.exit("file format not supported (try csv, parquet or fits)")
    else:
        sys.exit("file does not exist")
    print(f"file contains {str(len(star_data))} stars")
    return star_data



def ComputeInsolation_obs(Rs, a_AU, Teff):
    """
    INPUT:
        Rs: stellar radius [m]
        a_AU: semi major axis of the planet [AU]
        Teff: stellar effective temperature [K]
    OUTPUT:
        lum: stellar luminosity [L_sun]
        insol: insolation received by the planet [I_sun]
    """
    #lum = (Rs**2.)*((Teff/5778)**4.) #which units??
    ## tout en SI
    lum = 4*np.pi*(Rs**2.)*(Teff**4.)*cst.sigma_sb.value
    lum = lum / cst.L_sun.value
    #insol = lum*((1.*u.au/a)**2.)
    insol = lum / ((a_AU)**2.)
    return lum, insol

def calculate_star_radius_externe(df_stars, col_logl='logl', col_teff='teff'):
    """
    compute stellar radius
    INPUT:
        columns needed: star_teff [K]: can be either a single value or a df column
                        star_l: can be either a single value or a df column
                                assumed to be in unit of L_sun

    OUTPUT:
        return stellar radius based on Stefan Boltzmann law: L = 4*pi * R^2 * sigma_sb * T_eff^4
        in Rsun units
    """
        
    luminosity = 10**(df_stars[col_logl])*cst.L_sun.value
    star_teff = df_stars[col_teff]
    return (np.sqrt(luminosity /(4*np.pi*cst.sigma_sb.value))/(star_teff**2))*u.m.to(u.Rsun)

def calculateEffectiveFluxBoundary_plot(teff_K):
    """ 
    for plot purposes
    ---------------
    Use to get the two boundary lines of HZ. Obtained from https://github.com/fenrir-lin/exoPlot. 
    Formula from https://www.annualreviews.org/doi/10.1146/annurev-astro-082214-122238 
    """
    # inner limit: recent Venus
    SUN_IN = 1.7665
    A_IN = 1.3351E-4
    B_IN = 3.1515E-9
    C_IN = -3.3488E-12
    # outer limit: early Mars
    SUN_OUT = 0.324
    A_OUT = 5.3221E-5
    B_OUT = 1.4288E-9
    C_OUT = -1.1049E-12
    T = teff_K - 5780 # Teff and T are in K
    Seff_in = SUN_IN + A_IN*T + B_IN*(T**2) + C_IN*(T**3)
    Seff_out = SUN_OUT + A_OUT*T + B_OUT*(T**2) + C_OUT*(T**3)
    # Seff in unit of S_sun
    return Seff_in, Seff_out


### fonction for plots

def plot_hist_e(ax, data, Pmin, Pmax, in_years=True, display_in_days=True, 
                color_hist='tab:blue', density=True, orientation_hist='vertical'):
    """
    ax: where to plot
    data: df
    P_min/max: boundary values of period for wich we want eccentricity histogram 
    (by default in yr: otherwise set in_years to False)
    display_in_days: default==True, unit of the legend
    """
    
    if in_years==False:
        Pmin_label = Pmin
        Pmax_label = Pmax
        Pmin = Pmin*u.day.to(u.yr)
        Pmax = Pmax*u.day.to(u.yr)
        if display_in_days:
            if Pmax_label >= 1e3:
                Pmin_label = '%.e' % Pmin_label
                Pmax_label = '%.e' % Pmax_label
            label_hist = "%s < P < %s days"%(Pmin_label, Pmax_label)
        else:
            label_hist = "%s < P < %s years"%(np.round(Pmin, 3), np.round(Pmax, 3))
    else:
        if display_in_days:
            label_hist = "%s < P < %s days"%(np.round(Pmin*u.yr.to(u.day),0), np.round(Pmax*u.yr.to(u.day),0))
        else:
            label_hist = "%s < P < %s years"%(Pmin, Pmax)

    query_str = "period_yr > %s & period_yr < %s"%(Pmin, Pmax)
    return ax.hist(data.query(query_str).excentricity.values, density=density, bins='auto', 
                   histtype='step', color=color_hist, lw=3, orientation=orientation_hist, label=label_hist)

def plot_P_interval(ax, Pmin, Pmax, in_years=True, display_in_days=True, color='tab:blue', alpha_value=0.4):
    """
    plot P interval corresponding to an excentricity histogram
    """
    if in_years==False: # in days
        # if display_in_days: ok, do nothing
        if display_in_days==False: 
            # display in years
            Pmin = Pmin*u.day.to(u.yr)
            Pmax = Pmax*u.day.to(u.yr)
    else: # in years
        if display_in_days:
            Pmin = Pmin*u.yr.to(u.day)
            Pmax = Pmax*u.yr.to(u.day)
        # else ok, do nothing

    return ax.axvspan(Pmin, Pmax, alpha=alpha_value, color=color, zorder=2)

def convert_ax_days_to_years(ax1, ax_y):
    """
    Update second axis according with first axis.
    """
    y1, y2 = ax1.get_xlim()
    ax_y.set_xlim(y1*u.day.to(u.yr), y2*u.day.to(u.yr))
    ax_y.figure.canvas.draw()

def convert_days_to_years(Pdays):
    """
    convert Periods in days to years
    """
    Pyears = Pdays*u.day.to(u.yr)
    return Pyears

def convert_years_to_days(Pyears):
    """
    convert Periods in years to days
    """
    Pdays = Pyears*u.yr.to(u.day)
    return Pdays


def sig_radinf10(rad):
    return 0.08 * rad**3.57

def sig_radinf138(rad):
    return 0.15 * rad**1.49

def sig_radsup138(rad):
    return 4 * rad**100

def Convert_pl_radius_to_mass_realistic_custom(df_pl, mass_col='mass_planet', radius_col='radius_planet', simple=False):
    """
    Convert planet masses (in Mearth) in planet radius (in Rearth),
    based on relation derived by L. Parc+24
    dispersion det by eye
    17/01/25: para los planetas entre los 2 primeros slopes, randomly det à laquelle appartiennent 
    """

    df_pl[mass_col] = np.zeros(len(df_pl))
    index_segment1 = df_pl[df_pl[radius_col] <= 1.9].index
    index_segment2 = df_pl[(df_pl[radius_col] > 2.1) & (df_pl[radius_col] <= 13)].index
    # randomly distribute the ones between 7 and 15 Mearth
    in_between = df_pl[(df_pl[radius_col] > 1.9) & (df_pl[radius_col] <= 2.1)]
    bottom = in_between.sample(frac=0.5)
    top = in_between.loc[~in_between.index.isin(bottom.index)]
    index_segment1 = index_segment1.append(bottom.index)
    index_segment2 = index_segment2.append(top.index)
    index_segment3 = df_pl[df_pl[radius_col] > 13].index

    ### param mean and std for each segment
    mu_inf10 = 0.98 * df_pl.loc[index_segment1, radius_col]**3.57
    sig_inf10 = sig_radinf10(df_pl.loc[index_segment1, radius_col].values) #np.ones(len(primary_inf10))*0.1
    binf_inf10 = np.ones(len(index_segment1))*0.1
    bsup_inf10 = np.ones(len(index_segment1))*14

    #print((binf_inf10-mu_inf10)/sig_inf10)
    if simple:
        R_inf10 = mu_inf10
    else:
        R_inf10 = st.truncnorm.rvs(a=np.array((binf_inf10-mu_inf10)/sig_inf10, dtype='float'), b=np.array((bsup_inf10-mu_inf10)/sig_inf10, dtype='float'), 
                                loc=mu_inf10, scale=sig_inf10)
        
    mu_inf138 = 1.64 * df_pl.loc[index_segment2, radius_col]**1.49
    sig_inf138 = sig_radinf138(df_pl.loc[index_segment2, radius_col].values) #np.ones(len(primary_inf10))*0.1
    binf_inf138 = np.ones(len(index_segment2))*5
    bsup_inf138 = np.ones(len(index_segment2))*138
    if simple:
        R_inf138 = mu_inf138
    else:
        R_inf138 = st.truncnorm.rvs(a=np.array((binf_inf138-mu_inf138)/sig_inf138, dtype='float'), b=np.array((bsup_inf138-mu_inf138)/sig_inf138, dtype='float'), 
                                loc=mu_inf138, scale=sig_inf138)

    mu_sup138 = 10**(-45) * df_pl.loc[index_segment3, radius_col]**50
    sig_sup138 = sig_radsup138(df_pl.loc[index_segment3, radius_col].values) #np.ones(len(primary_inf10))*0.1
    binf_sup138 = np.ones(len(index_segment3))*138
    bsup_sup138 = np.ones(len(index_segment3))*10000
    if simple:
        R_sup138 = mu_sup138
        #print(R_sup138)
        #return mu_sup138
        R_sup138.loc[mu_sup138[mu_sup138>1000].index.values] = 10**np.random.uniform(2, 3, len(mu_sup138[mu_sup138>1000].index.values))
        #print(R_sup138)

    else:
        R_sup138 = st.truncnorm.rvs(a=np.array((binf_sup138-mu_sup138)/sig_sup138, dtype='float'), b=np.array((bsup_sup138-mu_sup138)/sig_sup138, dtype='float'), 
                                loc=mu_sup138, scale=sig_sup138)

    df_pl.loc[index_segment1, mass_col] = R_inf10
    df_pl.loc[index_segment2, mass_col] = R_inf138
    df_pl.loc[index_segment3, mass_col] = R_sup138

    return df_pl[mass_col]

### plot fraction of != planet types as bars
def plot_bars(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = ['#251FC1', '#B865AF', '#F17604']
    #['mediumblue', 'mediumorchid', 'darkorange']

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        #r, g, b, _ = color
        text_color = 'white' #'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color, size=13, weight="bold", padding=-10)
    ax.legend(bbox_to_anchor=(0, 1), ncol=len(category_names),
              loc='lower left', fontsize=13)
    ax.tick_params(axis='y', which='major', labelsize=13)

    return fig, ax


def calcul_semi_mayor_axis(df_plnts):
    """
    compute a for a given set of planets
    using Kepler 3rd law
    P in seconds (converted from years)
    M in kg (converted from Msun)
    returns a in AU (calculated in m)
    """
    ############################## mass etoile ptn pas planete......
    a_m = (cst.G.value * (df_plnts.mass_planet*u.Mearth.to(u.kg) + df_plnts.mass_star*u.Msun.to(u.kg)) / (4*np.pi**2))**(1./3.) * (df_plnts.period_planet*u.day.to(u.s))**(2./3.)
    return a_m*u.m.to(u.AU)  


### fct to re run planet creation (from already computed stellar pop)
def rerun_planetcreation(sample_name, fecha, save_pl=False, new_name='rerun_xxmes24_y'):
    """
    allows to re compute planet creation from a stellar population of single stars
    """
    print("######################################################")
    print("####### reruning planet creation for:", sample_name)
    ### initialize class object
    sample_stars = StarSample()

    ### load singles stars data
    sample_stars.load_previously_computed_files(sample_name, fecha, load_singles = True, load_singles_with_planetcols=False)

    ### re run planet creation and save planet stats with singles
    path_to_save_planets = "data/" + sample_name + "_PLANETS_" + fecha + ".parquet"
    path_to_save_singles_wth_planetnbs = "data/" + sample_name + "_SINGLES_and_PLANETSstats_" + fecha + ".parquet"
    sample_stars.create_planets(path_to_save_planets, path_to_save_singles_wth_planetnbs, save_pl=False, new_name=new_name)

    print("######## cleaning memory... (?)")
    sample_stars = None



###################################################
##### planet mass to radius conversion ############
###################################################

def sig_massinf10(masses):
    return 0.08 * masses**0.28

def sig_massinf138(masses):
    return 0.15 * masses**0.67

def sig_masssup138(masses):
    return 4 * masses**0.01

def Convert_pl_mass_to_radius_realistic(df_pl):
    """
    Convert planet masses (in Mearth) in planet radius (in Rearth),
    based on relation derived by L. Parc+24
    dispersion det by eye
    17/01/25: para los planetas entre los 2 primeros slopes, randomly det à laquelle appartiennent 
    """

    df_pl['radius_planet'] = np.zeros(len(df_pl))
    index_segment1 = df_pl[df_pl['mass_planet'] <= 5].index
    index_segment2 = df_pl[(df_pl['mass_planet'] > 14) & (df_pl['mass_planet'] <= 138)].index
    # randomly distribute the ones between 7 and 15 Mearth
    in_between = df_pl[(df_pl['mass_planet'] > 5) & (df_pl['mass_planet'] <= 14)]
    bottom = in_between.sample(frac=0.5)
    top = in_between.loc[~in_between.index.isin(bottom.index)]
    index_segment1 = index_segment1.append(bottom.index)
    index_segment2 = index_segment2.append(top.index)
    index_segment3 = df_pl[df_pl['mass_planet'] > 138].index

    ### param mean and std for each segment
    mu_inf10 = 1.02 * df_pl.loc[index_segment1, 'mass_planet']**0.28
    sig_inf10 = sig_massinf10(df_pl.loc[index_segment1, 'mass_planet'].values) #np.ones(len(primary_inf10))*0.1
    binf_inf10 = np.ones(len(index_segment1))*0.1
    bsup_inf10 = np.ones(len(index_segment1))*3
    R_inf10 = st.truncnorm.rvs(a=(binf_inf10-mu_inf10)/sig_inf10, b=(bsup_inf10-mu_inf10)/sig_inf10, 
                               loc=mu_inf10, scale=sig_inf10)
        
    mu_inf138 = 0.61 * df_pl.loc[index_segment2, 'mass_planet']**0.67
    sig_inf138 = sig_massinf138(df_pl.loc[index_segment2, 'mass_planet'].values) #np.ones(len(primary_inf10))*0.1
    binf_inf138 = np.ones(len(index_segment2))*2
    bsup_inf138 = np.ones(len(index_segment2))*20
    R_inf138 = st.truncnorm.rvs(a=(binf_inf138-mu_inf138)/sig_inf138, b=(bsup_inf138-mu_inf138)/sig_inf138, 
                                loc=mu_inf138, scale=sig_inf138)

    mu_sup138 = 11.9 * df_pl.loc[index_segment3, 'mass_planet']**0.01
    sig_sup138 = sig_masssup138(df_pl.loc[index_segment3, 'mass_planet'].values) #np.ones(len(primary_inf10))*0.1
    binf_sup138 = np.ones(len(index_segment3))*8
    bsup_sup138 = np.ones(len(index_segment3))*25
    R_sup138 = st.truncnorm.rvs(a=(binf_sup138-mu_sup138)/sig_sup138, b=(bsup_sup138-mu_sup138)/sig_sup138, 
                                loc=mu_sup138, scale=sig_sup138)

    df_pl.loc[index_segment1, 'radius_planet'] = R_inf10
    df_pl.loc[index_segment2, 'radius_planet'] = R_inf138
    df_pl.loc[index_segment3, 'radius_planet'] = R_sup138

    return df_pl['radius_planet']

def Convert_pl_mass_to_radius_realistic_custom(df_pl, mass_col='mass_planet', radius_col='radius_planet'):
    """
    Convert planet masses (in Mearth) in planet radius (in Rearth),
    based on relation derived by L. Parc+24
    dispersion det by eye
    17/01/25: para los planetas entre los 2 primeros slopes, randomly det à laquelle appartiennent 
    """

    df_pl[radius_col] = np.zeros(len(df_pl))
    index_segment1 = df_pl[df_pl[mass_col] <= 5].index
    index_segment2 = df_pl[(df_pl[mass_col] > 14) & (df_pl[mass_col] <= 138)].index
    # randomly distribute the ones between 7 and 15 Mearth
    in_between = df_pl[(df_pl[mass_col] > 5) & (df_pl[mass_col] <= 14)]
    bottom = in_between.sample(frac=0.5)
    top = in_between.loc[~in_between.index.isin(bottom.index)]
    index_segment1 = index_segment1.append(bottom.index)
    index_segment2 = index_segment2.append(top.index)
    index_segment3 = df_pl[df_pl[mass_col] > 138].index

    ### param mean and std for each segment
    mu_inf10 = 1.02 * df_pl.loc[index_segment1, mass_col]**0.28
    sig_inf10 = sig_massinf10(df_pl.loc[index_segment1, mass_col].values) #np.ones(len(primary_inf10))*0.1
    binf_inf10 = np.ones(len(index_segment1))*0.1
    bsup_inf10 = np.ones(len(index_segment1))*3
    R_inf10 = st.truncnorm.rvs(a=np.array((binf_inf10-mu_inf10)/sig_inf10, dtype='float'), b=np.array((bsup_inf10-mu_inf10)/sig_inf10, dtype='float'), 
                               loc=mu_inf10, scale=sig_inf10)
        
    mu_inf138 = 0.61 * df_pl.loc[index_segment2, mass_col]**0.67
    sig_inf138 = sig_massinf138(df_pl.loc[index_segment2, mass_col].values) #np.ones(len(primary_inf10))*0.1
    binf_inf138 = np.ones(len(index_segment2))*2
    bsup_inf138 = np.ones(len(index_segment2))*20
    R_inf138 = st.truncnorm.rvs(a=np.array((binf_inf138-mu_inf138)/sig_inf138, dtype='float'), b=np.array((bsup_inf138-mu_inf138)/sig_inf138, dtype='float'), 
                                loc=mu_inf138, scale=sig_inf138)

    mu_sup138 = 11.9 * df_pl.loc[index_segment3, mass_col]**0.01
    sig_sup138 = sig_masssup138(df_pl.loc[index_segment3, mass_col].values) #np.ones(len(primary_inf10))*0.1
    binf_sup138 = np.ones(len(index_segment3))*8
    bsup_sup138 = np.ones(len(index_segment3))*25
    R_sup138 = st.truncnorm.rvs(a=np.array((binf_sup138-mu_sup138)/sig_sup138, dtype='float'), b=np.array((bsup_sup138-mu_sup138)/sig_sup138, dtype='float'), 
                                loc=mu_sup138, scale=sig_sup138)

    df_pl.loc[index_segment1, radius_col] = R_inf10
    df_pl.loc[index_segment2, radius_col] = R_inf138
    df_pl.loc[index_segment3, radius_col] = R_sup138

    return df_pl[radius_col]


###### planetary systems study
def create_planetary_systems_custom(sample_name, fecha, path_saving_planets):
    """
    INPUT:
        sample_name, fecha: param for file names
        path_saving_planets: path to existing df with all synthetic planet population
        path_saving_planetary_systems: path where to save planetary syst df
    GOAL:
    - group planets in their planetary systems (1 line/syst)
    - compute coefficients Cv, Cs (defined by Mishra+23)
    """
    path_saving_planetary_systems = "data/" + sample_name + "_PLANETARY_SYST_" + fecha + ".parquet"
    if os.path.exists(path_saving_planetary_systems):
        print("Planetary systems already created, loaded into memory")
        planetary_systems = pd.read_parquet(path_saving_planetary_systems)
    else:
        print("### creation of planetary system df...")
        ### load planets df 
        planets = pd.read_parquet(path_saving_planets)
        print("# sort planets (may take around a minute...)")
        starttime = timer()
        ### sort planets by planetary systems, and inside planetary systems by distance to their star
        sorted_planets_temp = planets.sort_values(by=['id_hoststar', 'a_AU'])
        print("took", timer()-starttime)
        ### group by planetary systems
        pl_syst_temp = sorted_planets_temp.groupby(['id_hoststar'])
        print("calculate Cv and Cs coefficients..")
        ### compute Cv and Cs coefficients defined by Mishra+23
        Cv = pl_syst_temp.mass_planet.std()/pl_syst_temp.mass_planet.mean()
        Cv.rename('Cv_Mpl', inplace=True)
        n = pl_syst_temp.size()
        Cs = (np.log10(pl_syst_temp.mass_planet.prod()/pl_syst_temp.mass_planet.first()) - np.log10(pl_syst_temp.mass_planet.prod()/pl_syst_temp.mass_planet.last())) / (n - 1)
        Cs.rename('Cs_Mpl', inplace=True)
        ### we create a new df for the planetary systems... 
        # we want the columns: 'id_hoststar', 'mass_star', 'radius_star', 'logl_star', 'teff_star', 'mh_star', 'age_star', 'nb_planets', 'Cs_Mpl', 'Cv_Mpl'])
        # copy id_hoststar and star properties..
        #self.planetary_systems = pl_syst_temp[['id_hoststar', 'mass_star', 'radius_star', 'logl_star', 'teff_star','mh_star', 'age_star']].first().copy()
        planetary_systems = pl_syst_temp[['id_hoststar', 'mass_star', 'radius_star', 'teff_star']].first().copy()
        
        planetary_systems.reset_index(drop=True, inplace=True)
        # add nb of planets
        planetary_systems['nb_planets'] = pl_syst_temp['mass_planet'].count().values
        planetary_systems['total_pl_mass'] = pl_syst_temp['mass_planet'].sum().values
        planetary_systems['Cs_Mpl'] = Cs.values
        planetary_systems['Cv_Mpl'] = Cv.values
        # classify planetary systems into the 4 categories of Mishra
        planetary_systems["plsyst_type"] = np.where(planetary_systems['Cs_Mpl'] < -0.2, 'Antiordered', 
                 np.where(planetary_systems['Cs_Mpl'] > 0.2, 'Ordered', 
                          np.where(planetary_systems['Cv_Mpl'] <= (np.sqrt(planetary_systems['nb_planets']-1)/2), 'Similar', 
                               'Mixed')))
        print("## DONE...")
        #planetary_systems.to_parquet(path_saving_planetary_systems)
        #print("## ... and saved!")
        return planetary_systems
