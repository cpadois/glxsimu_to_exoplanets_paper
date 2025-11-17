import configparser
import os.path
import sys
import ast
from src.modules import occ_rates_fcts as OCC_RATE_MODULE
from src.modules import detectability_fcts as DETECTABILITY_MODULE
from src.modules import exoplanet_parameters_fcts as EXOPL_PARAMS_MODULE
from src.modules import glxsimu_to_stars_fcts as STAR_SIMU_MODULE
 
def get_params(param_file):
    """ reads parameter file and returns parameter dictionary """
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read(param_file)
    params = {s:dict(config.items(s)) for s in config.sections()}
    return params

def write_params(params, param_file):
    """ saves parameter dictionary as a parameter file """
    config = configparser.RawConfigParser()
    for sec in list(params.keys()):
        config.add_section(sec)
        for key in list(params[sec].keys()):
            config.set(sec, key, params[sec][key])
    cfgfile = open(param_file, 'w')
    config.write(cfgfile)
    cfgfile.close()
    print("Created new parameter file ", param_file)

def check_extra_st_params_field(config_file):
    if config_file['COLUMNS']['extra_stellar_params'] != '':
        extracted_extraparams_list = ast.literal_eval(config_file['COLUMNS']['extra_stellar_params'])
        extracted_extraparams_list = [i.strip() for i in extracted_extraparams_list]
        config_file['COLUMNS']['extra_stellar_params'] = extracted_extraparams_list
    else:
        pass
    return config_file

def get_columns(config_file):
    """ reads config file and returns dictionary with column names of the star input catalog"""
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read(config_file)
    col_dict = {s:dict(config.items(s)) for s in config.sections()}
    col_dict = check_extra_st_params_field(col_dict)
    return col_dict

def check_input_params(params):
    """
    check for missing parameters and change format (bools etc)
    """
    ######## check if dict not empty
    if params == {}:
        #print("Error: empty dictionary...")
        sys.exit("Empty dictionary")
    ######## convert all boolean like fields to actual booleans
    if not isinstance(params['INPUT_STARS']['simulate_stars'], bool):
        params['INPUT_STARS']['simulate_stars'] = params['INPUT_STARS']['simulate_stars'].lower() in ['true', '1', 't', 'y', 'yes']
    if not isinstance(params['INPUT_PLANET_CREATION']['simulate_planets'], bool):
        params['INPUT_PLANET_CREATION']['simulate_planets'] = params['INPUT_PLANET_CREATION']['simulate_planets'].lower() in ['true', '1', 't', 'y', 'yes']
    if not isinstance(params['INPUT_PLANET_CREATION']['save_planetary_systems'], bool):
        params['INPUT_PLANET_CREATION']['save_planetary_systems'] = params['INPUT_PLANET_CREATION']['save_planetary_systems'].lower() in ['true', '1', 't', 'y', 'yes']
    if not isinstance(params['INPUT_DETECTION']['simulate_detectability'], bool):
        params['INPUT_DETECTION']['simulate_detectability'] = params['INPUT_DETECTION']['simulate_detectability'].lower() in ['true', '1', 't', 'y', 'yes']

    #######################################################################
    # star simulation: if False, use star input file 
    #######################################################################
    if params['INPUT_STARS']['simulate_stars']:
        ### check if galactic_simu and galactic_region are correct
        if (params['INPUT_STARS']['galactic_simu'] == '') or (params['INPUT_STARS']['galactic_region'] == ''):
            print("Please specify the galactic simulation you want to use + the region where to select the stellar particles")
            print("> options for galactic_simu: 'NIHAO_UHD_g696e11', 'NIHAO_UHD_g696e11', 'NIHAO_UHD_g708e11', 'NIHAO_UHD_g755e11', 'NIHAO_UHD_826e11', 'NIHAO_UHD_g112e12', 'NIHAO_UHD_g279e12'")
            print("> options for galactic_region: 'SN', 'center', 'inner', 'outer', 'upper', 'lower', 'phi60', 'phi300'")
            #check_default = input("Assign default ('g755e11' and 'SN'), do you want to proceed? [y/n]")
            check_default = 'y'
            if check_default != 'y':
                sys.exit("User ended the process")
            else:
                params['INPUT_STARS']['galactic_simu'] = 'NIHAO_UHD_g755e11'
                params['INPUT_STARS']['galactic_region'] = 'SN'
        ### check if imf field is filled (or if value exists??), if not assign default
        if params['INPUT_STARS']['imf'] not in STAR_SIMU_MODULE.available_imf:
            print("IMF not specified or not supported, assigned to default: 'kirkpatrick24'")
            params['INPUT_STARS']['imf'] = 'kirkpatrick24'
        if not os.path.exists(params['INPUT_STARS']['path_isochrones_file']):
            print(f"isochrones file not found ({params['INPUT_STARS']['path_isochrones_file']}): please change path or download the file at xxxx")
            sys.exit("isochrones file not found...")
    else:
        ### check if path_star_file AND star_config are given
        if (params['INPUT_STARS']['path_star_file'] == '') or (params['INPUT_STARS']['star_config'] == ''):
            print("Please give an input star table + config file with column names")
        else:
            if not os.path.exists(params['INPUT_STARS']['path_star_file']):
                print(f"Invalid path or invalid star table ({params['INPUT_STARS']['path_star_file']})...")
            ### also check config file here??
    #######################################################################
    # planets simulation
    #######################################################################
    if params['INPUT_PLANET_CREATION']['simulate_planets']:
        if params['INPUT_PLANET_CREATION']['custom_occ_rate'] not in OCC_RATE_MODULE.available_occrates:
            print("Occurrence rate model not valid or not yet available, set to default ('M_Burn21_Feh_Narang18_A')")
            params['INPUT_PLANET_CREATION']['custom_occ_rate'] = 'M_Burn21_Feh_Narang18_A'
        if params['INPUT_PLANET_CREATION']['mass_period_distrib'] not in EXOPL_PARAMS_MODULE.available_MP:
            print("Mass-Period distribution model not valid or not yet available, set to default ('initial_4gaussians')")
            params['INPUT_PLANET_CREATION']['mass_period_distrib'] = 'initial_4gaussians'
        if params['INPUT_PLANET_CREATION']['photoevap_effect'] != '':
            print("Photoevaporation effect not yet implemented... coming one day")
            params['INPUT_PLANET_CREATION']['photoevap_effect'] = ''
        if params['INPUT_PLANET_CREATION']['eccentricity_distrib'] != '':
            print("Eccentricity distribution not yet implemented... coming one day")
            params['INPUT_PLANET_CREATION']['eccentricity_distrib'] = ''

    #######################################################################
    # detectability simulation
    #######################################################################
    if params['INPUT_DETECTION']['simulate_detectability']:
        if not params['INPUT_PLANET_CREATION']['simulate_planets']:
            print("Error, no planet to be detected, set 'simulate_detectability' to False")
            params['INPUT_DETECTION']['simulate_detectability'] = False
        if params['INPUT_DETECTION']['observer'] not in DETECTABILITY_MODULE.available_detectors:
            print("Detection simulation for this instrument is not valid or not yet available, set to default ('Kepler')")
            params['INPUT_PLANET_CREATION']['observer'] = 'Kepler'
    
    return params


def check_config_star(col_dict, starfile_columns):
    """
    verify presence of mandatory columns (M and [Fe/H])
    """
    ######## check if dict not empty
    if col_dict == {}:
        #print("Error: empty dictionary...")
        sys.exit("Empty dictionary")
    ### mandatory columns
    if col_dict['COLUMNS']['column_stellar_mass'] not in starfile_columns:
        print("Error: Missing or incorrect stellar mass column name, correct in config file...")
        sys.exit("(can't proceed to planet simulation without)")

    if col_dict['COLUMNS']['column_stellar_metallicity'] not in starfile_columns:
        print("Missing or incorrect stellar metallicity column name, correct in config file...")
        sys.exit("(can't proceed to planet simulation without)")

    ### optional columns 
    # (if they are given, check if they exist in table, if no set to default)
    print_warning = False
    for col in ['column_teff', 'column_luminosity', 'column_stellar_radius']:
        if col_dict['COLUMNS'][col] not in starfile_columns:
            print_warning = True
            print("%s not found in input star table, set to default (empty)"%(col))
            col_dict['COLUMNS'][col] = ''
    if print_warning:
        print("(exoplanet generation will proceed but some functions will not be available)")

    if col_dict['COLUMNS']['extra_stellar_params'] != '':
        try:
            try:
                list_columns = ast.literal_eval(col_dict['COLUMNS']['extra_stellar_params'])
                list_columns = [i.strip() for i in list_columns]
            except:
                list_columns = col_dict['COLUMNS']['extra_stellar_params'] # already a proper list
            for ecol in list_columns:
                if ecol not in starfile_columns:
                    print("Column %s not found in input star table, set to default (empty)"%(ecol))
                    print("(exoplanet generation will proceed)")
        except:
            print("extra_stellar_params field seems to not have the correct format (list of str), ignored and proceed without")
            col_dict['COLUMNS']['extra_stellar_params'] = ''

    return col_dict
