"""
creation: 28/08/25

Contains fct(s) descriving the occurrence rate of different panet types, 
as a fct of stellar mass and metallicity.

- M_Burn21_Feh_Narang18_A
"""

import numpy as np

#####################################################################
# available occ rate models
#####################################################################
available_occrates = ['M_Burn21_Feh_Narang18_A']

def choice_occrate_fct(custom_occrate_name):
    if custom_occrate_name == 'M_Burn21_Feh_Narang18_A':
        occ_rate_Earthlike_MFeh = occ_rate_Earthlike_Aboth
        occ_rate_SEnept_MFeh = occ_rate_SEnept_Aboth
        occ_rate_giants_MFeh = occ_rate_giants_Aleft

    else:
        print("occurrence rate model not existing or not implemented yet")

    return [occ_rate_Earthlike_MFeh, occ_rate_SEnept_MFeh, occ_rate_giants_MFeh]


def choice_multiplicity_fct(custom_multiplicity_name):
    if custom_multiplicity_name == 'M_Burn21':
        multiplicity_Earthlike = multiplicity_Earthlike_Burn21
        multiplicity_SE_nept = multiplicity_SE_nept_Burn21
        multiplicity_giants = multiplicity_giants_Burn21

    else:
        print("multiplicity model not existing or not implemented yet")

    return [multiplicity_Earthlike, multiplicity_SE_nept, multiplicity_giants]
################################################################################################
######### number of exoplanet of each type / occ.rate ##########################################
################################################################################################

###### MASS DEPENDENCE

stellar_masses_Burn21 = np.array([0.1, 0.3, 0.5, 0.7, 1.0])

### fractions from Burn+21 for 0.1, 0.3, 0.5, 0.7, 1.0 M_sun stars
# Earth-like
fraction_stars_with_earthlike = np.array([0.7, 0.88, 0.89, 0.89, 0.84])

# Super-Earth and Neptunian combined (fractions summed)
frac_SE = np.array([0.19, 0.54, 0.71, 0.78, 0.79])
frac_nept = np.array([0.01, 0.08, 0.17, 0.22, 0.27])
fraction_stars_with_superearth_nept = np.ones(len(stellar_masses_Burn21))
for i in range(len(stellar_masses_Burn21)):
    fraction_stars_with_superearth_nept[i] = frac_SE[i]+frac_nept[i] - (frac_SE[i]*frac_nept[i])

# sub-giants and giants combined (fractions summed)
frac_subg = np.array([0.0, 0.0, 0.02, 0.03, 0.05])
frac_g = np.array([0.0 ,0.0, 0.02, 0.09, 0.19])
fraction_stars_with_giants = np.ones(len(stellar_masses_Burn21))
for i in range(len(stellar_masses_Burn21)):
    fraction_stars_with_giants[i] = frac_subg[i]+frac_g[i] - (frac_subg[i]*frac_g[i])

### interpolation
def fraction_earthlike(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, fraction_stars_with_earthlike)

def fraction_SE_nept(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, fraction_stars_with_superearth_nept)

def fraction_giants(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, fraction_stars_with_giants)

### multiplicity of specific planet types form Burn+21 for 0.1, 0.3, 0.5, 0.7, 1.0 M_sun stars
# Earth-like
multiplicity_earthlike_Burn = np.array([4.31, 5.58, 5.59, 5.14, 4.89])

# Super-Earth and Neptunian combined (fractions summed)
mult_SE = np.array([1.89, 3.23, 4.06, 4.44, 4.77])
mult_nept = np.array([1.0, 1.13, 1.33, 1.39, 1.33])
multiplicity_superearth_nept = np.ones(len(stellar_masses_Burn21))
for i in range(len(stellar_masses_Burn21)):
    multiplicity_superearth_nept[i] = mult_SE[i]+mult_nept[i]

# sub-giants and giants combined (fractions summed)
mult_subg = np.array([0.0, 1.0, 1.14, 1.06, 1.17])
mult_g = np.array([0.0, 1.0, 1.3, 1.58, 1.63])
multiplicity_s_giants = np.ones(len(stellar_masses_Burn21))
for i in range(len(stellar_masses_Burn21)):
    multiplicity_s_giants[i] = mult_subg[i]+mult_g[i]

### interpolation
def multiplicity_Earthlike_Burn21(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, multiplicity_earthlike_Burn)

def multiplicity_SE_nept_Burn21(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, multiplicity_superearth_nept)

def multiplicity_giants_Burn21(mass_star):
    return np.interp(mass_star, stellar_masses_Burn21, multiplicity_s_giants)


###### occurrence rates as a fct of STELLAR METALLICITY 
### values from Narang+18, fig.9 (extracted via WebPlotDigitizer...)
stellar_feh_Narang18 = np.array([-0.5, 0, 0.35]) # mean points of the 3 intervals

OR_1_2_Rearth = np.array([70.2193318754085, 57.0040236789636, 41.8839125445748])/100
OR_2_4_Rearth = np.array([43.0386130779573, 69.5856494710045, 40.760191943547])/100
OR_4_8_Rearth = np.array([4.54443938399406, 6.77187076526332, 11.1492099700809])/100
OR_8_20_Rearth = np.array([2.12212838078799, 7.21552135790102, 9.05092390032845])/100

### combine 2_4 and 4_8 in SE-nept category
OR_2_8_Rearth = np.ones(len(stellar_feh_Narang18))
for i in range(len(stellar_feh_Narang18)):
    OR_2_8_Rearth[i] = OR_2_4_Rearth[i]+OR_4_8_Rearth[i] - (OR_2_4_Rearth[i]*OR_4_8_Rearth[i])

def occ_rate_feh_Earthlike(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18, OR_1_2_Rearth)
def occ_rate_feh_SE_nept(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18, OR_2_8_Rearth)
def occ_rate_feh_giants(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18, OR_8_20_Rearth)

### combine occ rate metallicity and occ_rate mass
def occ_rate_Earthlike_plot2d(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    OR = (occ_rate_feh_Earthlike(FEH))*fraction_earthlike(M)/occ_rate_feh_Earthlike(0)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept_plot2d(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    return (occ_rate_feh_SE_nept(FEH))*fraction_SE_nept(M)/occ_rate_feh_SE_nept(0)
def occ_rate_giants_plot2d(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    return (occ_rate_feh_giants(FEH))*fraction_giants(M)/occ_rate_feh_giants(0)

def occ_rate_Earthlike(feh_star, mass_star):
    OR = (occ_rate_feh_Earthlike(feh_star)/occ_rate_feh_Earthlike(0))*fraction_earthlike(mass_star)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept(feh_star, mass_star):
    return (occ_rate_feh_SE_nept(feh_star)/occ_rate_feh_SE_nept(0))*fraction_SE_nept(mass_star)
def occ_rate_giants(feh_star, mass_star):
    return (occ_rate_feh_giants(feh_star)/occ_rate_feh_giants(0))*fraction_giants(mass_star)

######################################################################
######### alternative occ rates (!= metallicity dependence..)
stellar_feh_Narang18_A = np.array([-1.0, -0.5, 0, 0.35, 0.6])

OR_1_2_Rearth_A = np.array([0.0, 70.2193318754085, 57.0040236789636, 41.8839125445748, 0.0])/100
OR_2_4_Rearth_A = np.array([0.0, 43.0386130779573, 69.5856494710045, 40.760191943547, 0.0])/100
OR_4_8_Rearth_A = np.array([0.0, 4.54443938399406, 6.77187076526332, 11.1492099700809, 0.0])/100
OR_8_20_Rearth_A = np.array([0.0, 2.12212838078799, 7.21552135790102, 9.05092390032845, 0.0])/100

### combine 2_4 and 4_8 in SE-nept category
OR_2_8_Rearth_A = np.ones(len(stellar_feh_Narang18_A))
for i in range(len(stellar_feh_Narang18_A)):
    OR_2_8_Rearth_A[i] = OR_2_4_Rearth_A[i]+OR_4_8_Rearth_A[i] - (OR_2_4_Rearth_A[i]*OR_4_8_Rearth_A[i])


### alternative occ rate, on the left side
def occ_rate_feh_Earthlike_Aleft(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[:-1], OR_1_2_Rearth_A[:-1])
def occ_rate_feh_SE_nept_Aleft(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[:-1], OR_2_8_Rearth_A[:-1])
def occ_rate_feh_giants_Aleft(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[:-1], OR_8_20_Rearth_A[:-1])

### alternative occ rate, on the right side
def occ_rate_feh_Earthlike_Aright(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[1:], OR_1_2_Rearth_A[1:])
def occ_rate_feh_SE_nept_Aright(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[1:], OR_2_8_Rearth_A[1:])
def occ_rate_feh_giants_Aright(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A[1:], OR_8_20_Rearth_A[1:])

### alternative occ rate, both sides
def occ_rate_feh_Earthlike_Aboth(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A, OR_1_2_Rearth_A)
def occ_rate_feh_SE_nept_Aboth(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A, OR_2_8_Rearth_A)
def occ_rate_feh_giants_Aboth(feh_star):
    return np.interp(feh_star, stellar_feh_Narang18_A, OR_8_20_Rearth_A)

### combine w/ mass dependency
def occ_rate_Earthlike_Aleft(feh_star, mass_star):
    OR = (occ_rate_feh_Earthlike_Aleft(feh_star)/occ_rate_feh_Earthlike_Aleft(0))*fraction_earthlike(mass_star)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept_Aleft(feh_star, mass_star):
    return (occ_rate_feh_SE_nept_Aleft(feh_star)/occ_rate_feh_SE_nept_Aleft(0))*fraction_SE_nept(mass_star)
def occ_rate_giants_Aleft(feh_star, mass_star):
    return (occ_rate_feh_giants_Aleft(feh_star)/occ_rate_feh_giants_Aleft(0))*fraction_giants(mass_star)

def occ_rate_Earthlike_Aright(feh_star, mass_star):
    OR = (occ_rate_feh_Earthlike_Aright(feh_star)/occ_rate_feh_Earthlike_Aright(0))*fraction_earthlike(mass_star)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept_Aright(feh_star, mass_star):
    return (occ_rate_feh_SE_nept_Aright(feh_star)/occ_rate_feh_SE_nept_Aright(0))*fraction_SE_nept(mass_star)
def occ_rate_giants_Aright(feh_star, mass_star):
    return (occ_rate_feh_giants_Aright(feh_star)/occ_rate_feh_giants_Aright(0))*fraction_giants(mass_star)

def occ_rate_Earthlike_Aboth(feh_star, mass_star):
    OR = (occ_rate_feh_Earthlike_Aboth(feh_star)/occ_rate_feh_Earthlike_Aboth(0))*fraction_earthlike(mass_star)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept_Aboth(feh_star, mass_star):
    return (occ_rate_feh_SE_nept_Aboth(feh_star)/occ_rate_feh_SE_nept_Aboth(0))*fraction_SE_nept(mass_star)
def occ_rate_giants_Aboth(feh_star, mass_star):
    return (occ_rate_feh_giants_Aboth(feh_star)/occ_rate_feh_giants_Aboth(0))*fraction_giants(mass_star)

### and finally for illustrativ plot
### combine occ rate metallicity and occ_rate mass
def occ_rate_Earthlike_plot2d_A(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    OR = (occ_rate_feh_Earthlike_Aboth(FEH))*fraction_earthlike(M)/occ_rate_feh_Earthlike_Aboth(0)
    return np.where(OR>1, 1.0, OR)
def occ_rate_SEnept_plot2d_A(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    return (occ_rate_feh_SE_nept_Aboth(FEH))*fraction_SE_nept(M)/occ_rate_feh_SE_nept_Aboth(0)
def occ_rate_giants_plot2d_A(feh_star, mass_star):
    FEH,M = np.meshgrid(feh_star, mass_star)
    return (occ_rate_feh_giants_Aleft(FEH))*fraction_giants(M)/occ_rate_feh_giants_Aleft(0)
