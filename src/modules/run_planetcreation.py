"""
main module

need:
    - a parameter file (glxsimu_to_exoplanets_public.params for now)
    - if star table given in input, need of a config file with columns names
"""

import os.path
import sys
import numpy as np

from astropy import units as u
import astropy.constants as cst
import astropy.table
import pandas as pd

from tqdm import tqdm
from timeit import default_timer as timer

from src.modules import read_params
from src.modules import utils_star_sample_public as utils
from src.modules import glxsimu_to_stars_fcts as STARS_SIMU_MODULE


def main(param_file):
    start = timer()
    ################################################################################
    #----------------------- Get the input parameters ------------------------------
    ################################################################################
    print ("#"*42)
    print ("-------- import params ---------")
    print ("#"*42)
    print (" ")

    #if len(sys.argv) == 2:
    #    param_file = sys.argv[1]
    #    params = read_params.get_params(param_file)
    #elif len(sys.argv) == 1:
    #    print("Please give a parameter file in argument")
    #    exit
    #else:
    #    print ("Too many parameters given. Give me only the parameter file.")
    #    exit

    params = read_params.get_params(param_file)
    #### check params validity
    params = read_params.check_input_params(params)

    ################################################################################
    #----------------------- star simulation or input ------------------------------
    ################################################################################
    if params['INPUT_STARS']['simulate_stars']:
        print(">>>>>>> Simulation of stars from galactic simulation...")
        print("soon available... (only NIHAO-UHD for now)")
        str_star_table_origin = 'simulated stars'
        # StarSample()
        # create stars
        sample_stars =_simulate_stars(params)
        sample_planets = utils.PlanetSample()
        sample_planets.singles = sample_stars.singles
        
    else:
        sample_planets = utils.PlanetSample()
        print(">>>>>>> Import stars from input file %s"%(params['INPUT_STARS']['path_star_file']))
        sample_planets.singles = utils.read_star_file(params['INPUT_STARS']['path_star_file'])
        str_star_table_origin = 'user star catalog'
        print("### import column names from config file %s"%(params['INPUT_STARS']['star_config']))
        col_dict = read_params.get_columns(params['INPUT_STARS']['star_config'])
        col_dict = read_params.check_config_star(col_dict, sample_planets.singles.columns)
        sample_planets.extract_column_names(col_dict)

    ################################################################################
    #----------------------------- planet simulation -------------------------------
    ################################################################################
    if params['INPUT_PLANET_CREATION']['simulate_planets']:
        print(">>>>>>> Simulation of exoplanets around %s..."%(str_star_table_origin))
        sample_planets.create_planets_custom(params)
        if params['INPUT_PLANET_CREATION']['save_planetary_systems']:
            print(">>> creating a table with each planetary system and saving it")
            sample_planets.create_planetary_systems(params)

    if params['INPUT_DETECTION']['simulate_detectability']:
        if params['INPUT_DETECTION']['simulate_detectability'] != '':
            str_detector = params['INPUT_DETECTION']['observer']
        else:
            str_detector = '[no observer specified...]'
        print(">>>>>>> Simulation of detectability by %s"%(str_detector))
        # run detectability(params['INPUT_DETECTION'])
        print("soon available... (Kepler, PLATO, Roman, etc)")

    total_time = timer() - start
    print ("#"*42)
    print ("-------- FINISHED ---------")
    print (f"---- took {total_time} sec ----")
    print ("#"*42)


def _simulate_stars(params):
    """
    generate stars from the choosen galactic simulation
    """
    stars_simu = utils.StarSample()
    stellar_particles_df = STARS_SIMU_MODULE.select_stellar_particles(params).to_pandas()
    stars_simu.param_new_init(params, stellar_particles_df)
    stars_simu.creation_stars(params)
    stars_simu.make_multiple_systems(params)

    return stars_simu

