#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from perturb_helper_funcs import *


years = list(range(2009, 2019))
num_ensemble = 50

def perturb_soil_textures_and_parameters(iensemble=0):
    #Sand and clay content were perturbed with random noise drawn from spatially uniform distribution (±20 %). 
    #In order to avoid un-physical values of the soil parameters, the sum of the sand and clay content were 
    #constrained to have a value not larger than 100 %.
    sname = ("/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/surfdata_map/test_trial_22/SE-Svb/" +
             "ensemble/" +"surfdata_SE-Svb_hist_78pfts_CMIP6_simyr2000_c240511_pft_modified.nc_" +  
             str(iensemble + 1).zfill(5) + ".nc")
    sorig = ("/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/surfdata_map/test_trial_22/SE-Svb/ensemble/" +"surfdata_SE-Svb_hist_78pfts_CMIP6_simyr2000_c240511_pft_modified.nc")
    
    with nc.Dataset(sorig) as src, nc.Dataset(sname, "w") as dst:
        # Copy attributes
        copy_attr_dim(src, dst)
        # dimension of perturbed fields
        dim_lvl   = src.dimensions["nlevsoi"].size
        dim_lat   = src.dimensions["lsmlat"].size
        dim_lon   = src.dimensions["lsmlon"].size
        dim_types = 3

        # Perturb %SAND, %CLAY and OM:
        rnd_type_cell = np.random.uniform(low=-20.0, high=20.0, size=dim_lat*dim_lon*dim_types).reshape(dim_types, dim_lat*dim_lon)
        rnd = np.zeros((dim_types, dim_lvl, dim_lat*dim_lon))
        for t in range(dim_types):
            for c in range(dim_lat*dim_lon):
                rnd[t, :, c] = rnd_type_cell[t, c]
        rnd = rnd.reshape((dim_types, dim_lvl, dim_lat, dim_lon))

        # Keep percentages normalized (sum to 100)
        pct = np.array([src.variables["PCT_SAND"][:] + rnd[0],
                        src.variables["PCT_CLAY"][:] + rnd[1],
                        src.variables["ORGANIC"][:]  + rnd[2]])
                          
        # Keep in range between 0 and 99 percent 
        for l in range(dim_lvl):
            for la in range(dim_lat):
                for lo in range(dim_lon):
 #                   for t in range(dim_types):
                        if pct[0, l, la, lo] > 99.0: 
                            pct[0, l, la, lo] = 99.0
                        if pct[1, l, la, lo] > 99.0: 
                            pct[1, l, la, lo] = 99.0
                        if pct[0, l, la, lo] < 0.0: 
                            pct[0, l, la, lo] = 0.0
                        if pct[1, l, la, lo] < 0.0:
                            pct[1, l, la, lo] = 0.0
        # Keep OM in range 
                        if pct[2,l, la, lo] > 120.0: 
                             pct[2,l, la, lo] = 120.0
                        if pct[2,l, la, lo] < 0.0: 
                             pct[2,l, la, lo] = 0.0

        # Keep percentages normalized (sum to 100)       
        for l in range(dim_lvl):
            for la in range(dim_lat):
                for lo in range(dim_lon):
                    old_sum = np.sum(pct[:, l, la, lo])
                    for t in range(dim_types):
                        if old_sum > 100.0:
                            pct[:, l, la, lo] = 100.0 * pct[:, l, la, lo] / old_sum
        
        # After generating all random variables
        # save state of random number generator to file
        if not force_seed:
            rnd_state_serialize()

        # Copy non-perturbed variables:
        for name, var in src.variables.items():
            if name != "PCT_SAND" and name != "PCT_CLAY" and name != "ORGANIC":
                nvar = dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]

        
        # Add perturbations
        pct_sand = dst.createVariable("PCT_SAND", 
                                      datatype=np.float64, 
                                      dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                      fill_value=1.e+30)
        pct_sand.setncatts({'long_name': u"percent sand",
                            'units': u"unitless"})
        dst.variables["PCT_SAND"][:] = pct[0].reshape(dst.variables["PCT_SAND"].shape)
        SAND = dst.variables["PCT_SAND"][:]

        pct_clay = dst.createVariable("PCT_CLAY", 
                                      datatype=np.float64, 
                                      dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                      fill_value=1.e+30)
        pct_clay.setncatts({'long_name': u"percent clay",
                            'units': u"unitless"})
        dst.variables["PCT_CLAY"][:] = pct[1].reshape(dst.variables["PCT_CLAY"].shape)
        CLAY = dst.variables["PCT_CLAY"][:]
		
        om = dst.createVariable("ORGANIC", 
                                datatype=np.float64, 
                                dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                fill_value=1.e+30)
        om.setncatts({'long_name': u"organic matter density at soil levels",
                      'units': u"kg/m3 (assumed carbon content 0.58 gC per gOM)"})
        dst.variables["ORGANIC"][:] = pct[2].reshape(dst.variables["ORGANIC"].shape)
        
        
        # Perturb soil hydraulic parameters
        # Saturated soil matric potential
        psis_sat = dst.createVariable("PSIS_SAT",
                                    datatype=np.float64, 
                                    dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                    fill_value=1.e+30)
        psis_sat.setncatts({'long_name': u"Sat. soil matric potential",
                                'units': u"mmH20"})
        sucsat                       = 10. * ( 10.**(1.88-0.0131*SAND))
        sucsat_std                   = 0.72 + 0.0012*CLAY
        noise_sucsat                 = np.random.normal(loc=0.0, scale=sucsat_std, size=pct_sand.shape)
        perturbed_log_sucsat         = np.log(sucsat) + noise_sucsat
        back_transformed_sucsat      = np.clip(np.exp(perturbed_log_sucsat), 0, 1000)
        dst.variables["PSIS_SAT"][:] = back_transformed_sucsat

        # Porosity
        thetas = dst.createVariable("THETAS",
                                    datatype=np.float64, 
                                    dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                    fill_value=1.e+30)
        thetas.setncatts({'long_name': u"Porosity",
                                'units': u"vol/vol"})
        watsat                     = 0.489 - 0.00126*SAND
        watsat_std                 = (7.73-0.073*CLAY) / 100.0
        noise_watsat               = np.random.normal(loc=0.0, scale=watsat_std, size=pct_sand.shape)
        perturbed_watsat           = watsat + noise_watsat
        dst.variables["THETAS"][:] = perturbed_watsat

        # Shape (b) parameter
        shape_param = dst.createVariable("SHAPE_PARAM",
                                    datatype=np.float64, 
                                    dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                    fill_value=1.e+30)
        shape_param.setncatts({'long_name': u"Shape (b) parameter",
                                'units': u"unitless"})
        bsw                              = 2.91 + 0.159*CLAY
        bsw_std                          = 0.0500 * CLAY + 1.34 
        noise_bsw                        = np.random.normal(loc=0.0, scale=bsw_std, size=pct_clay.shape)
        perturbed_bsw                    = bsw + noise_bsw
        perturbed_bsw[perturbed_bsw < 0] = 0
        dst.variables["SHAPE_PARAM"][:]  = perturbed_bsw

        # Saturated hydraulic conductivity
        ks = dst.createVariable("KSAT",
                                datatype=np.float64, 
                                dimensions=("nlevsoi", "lsmlat", "lsmlon",), 
                                fill_value=1.e+30)
        ks.setncatts({'long_name': u"Sat. hydraulic conductivity", 'units': u"mm/s"})
        xksat                    = 0.0070556 *( 10.**(-0.884+0.0153*SAND))
        xksat_std                = 0.459 + 0.00321*(1-(SAND+CLAY)/100)
        noise_xksat              = np.random.normal(loc=0.0, scale=xksat_std, size=pct_sand.shape)
        perturbed_log_xksat      = np.log(xksat) + noise_xksat
        back_transformed_xksat   = np.exp(perturbed_log_xksat)
        dst.variables["KSAT"][:] = back_transformed_xksat
        
        #, bsw.flatten(), perturbed_bsw.flatten(), xksat.flatten(), back_transformed_xksat.flatten(), perturbed_xksat.flatten(), watsat.flatten(), perturbed_watsat.flatten(), sucsat, back_transformed_sucsat.flatten(), perturbed_sucsat.flatten()
        return SAND.flatten(), CLAY.flatten(), sucsat.flatten(), perturbed_log_sucsat.flatten(), back_transformed_sucsat.flatten(), watsat.flatten(), perturbed_watsat.flatten(), bsw.flatten(), perturbed_bsw.flatten(), xksat.flatten(), perturbed_log_xksat.flatten(), back_transformed_xksat.flatten()

rnd_state_file = "rnd_state.json"
force_seed = False 
# Either seed random number generator or continue with existing state
if not os.path.isfile(rnd_state_file) or force_seed:
    np.random.seed(22)
else:
    rnd_state_deserialize()

for ens in range(num_ensemble):
    perturb_soil_textures_and_parameters(ens)
    for y in years:
        print("Done with year " + str(y) + " ensemble " + str(ens))

## Create arrays to store data for all ensemble members
#all_SAND, all_CLAY, unpert_sucsat, log_perturbed_sucsat, back_sucsat, unpert_watsat, all_perturbed_watsat, unpert_bsw, all_perturbed_bsw, unpert_xksat, log_perturbed_xksat, back_xksat = [], [], [], [] , [], [], [], [], [], [], [], []
## Loop through ensemble members
#for ens in range(num_ensemble):
#    # Perturb soil properties and retrieve data
#    sand, clay, sucsat, perturbed_log_sucsat, back_transformed_sucsat, watsat, perturbed_watsat, bsw, perturbed_bsw, xksat, perturbed_log_xksat, back_transformed_xksat = perturb_soil_textures_and_parameters(ens)
#
#    # Append data to arrays
#    all_SAND.extend(sand)
#    all_CLAY.extend(clay)
#    unpert_sucsat.extend(sucsat)
#    log_perturbed_sucsat.extend(perturbed_log_sucsat)
#    back_sucsat.extend(back_transformed_sucsat)
#    unpert_watsat.extend(watsat)
#    all_perturbed_watsat.extend(perturbed_watsat)
#    unpert_bsw.extend(bsw)
#    all_perturbed_bsw.extend(perturbed_bsw)
#    unpert_xksat.extend(xksat)
#    log_perturbed_xksat.extend(perturbed_log_xksat)
#    back_xksat.extend(back_transformed_xksat)
#
## Plot soil parameter relationship to soil texture
#plt.figure(figsize=(20, 12))
#plt.rcParams.update({'font.size': 13})
#
#plt.subplot(4, 3, 1)
#plt.hist(unpert_sucsat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Suction (mm)')
#plt.ylabel('Frequency')
#plt.title('Suction Distribution')
#
## Plot perturbed log suction
#plt.subplot(4, 3, 2)
#plt.hist(log_perturbed_sucsat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Perturbed log Suction (mm)')
#plt.ylabel('Frequency')
#plt.title('Perturbed log Suction Distribution')
#
#plt.subplot(4, 3, 3)
#plt.scatter(all_SAND, log_perturbed_sucsat, marker='.', color='purple', alpha=0.5)
#plt.xlabel('%Sand')
#plt.ylabel('Sat. Suction (mm)')
#plt.title('Log Perturbed Suction and %Sand')
#
## Plot porosity without noise
#plt.subplot(4, 3, 4)
#plt.hist(unpert_watsat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Porosity (vol/vol)')
#plt.ylabel('Frequency')
#plt.title('Porosity Distribution')
#
## Plot perturbed porosity
#plt.subplot(4, 3, 5)
#plt.hist(all_perturbed_watsat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Perturbed Porosity (vol/vol)')
#plt.ylabel('Frequency')
#plt.title('Perturbed Porosity Distribution')
#
#plt.subplot(4, 3, 6)
#plt.scatter(all_SAND, all_perturbed_watsat, marker='.', color='green', alpha=0.5)
#plt.xlabel('%Sand')
#plt.ylabel('Porosity (vol/vol)')
#plt.title('Perturbed Porosity and %Sand')
#
## Plot shape parameter without noise
#plt.subplot(4, 3, 7)
#plt.hist(unpert_bsw, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Shape Parameter')
#plt.ylabel('Frequency')
#plt.title('Shape Parameter Distribution')
#
## Plot perturbed shape parameter
#plt.subplot(4, 3, 8)
#plt.hist(all_perturbed_bsw, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Perturbed Shape Parameter)')
#plt.ylabel('Frequency')
#plt.title('Perturbed Shape Parameter Distribution')
#
#plt.subplot(4, 3, 9)
#plt.scatter(all_CLAY, all_perturbed_bsw, marker='.', color='red', alpha=0.5)
#plt.xlabel('%Clay')
#plt.ylabel('Shape (b) parameter')
#plt.title('Perturbed Shape parameter and %Clay')
#
## Plot KSAT without noise
#plt.subplot(4, 3, 10)
#plt.hist(unpert_xksat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Ksat (mm/s)')
#plt.ylabel('Frequency')
#plt.title('Ksat Distribution')
#
## Plot perturbed shape parameter
#plt.subplot(4, 3, 11)
#plt.hist(log_perturbed_xksat, bins=20, color='gray', alpha=0.5, edgecolor='black')
#plt.xlabel('Perturbed log Ksat (mm/s)')
#plt.ylabel('Frequency')
#plt.title('Perturbed log Ksat Distribution')
#
#plt.subplot(4, 3, 12)
#plt.scatter(all_SAND, log_perturbed_xksat, marker='.', color='blue', alpha=0.5)
#plt.xlabel('%Sand')
#plt.ylabel('KSAT (mm/s)')
#plt.title('Log Perturbed KSAT and %Sand')
#
#plt.tight_layout()
#plt.show()