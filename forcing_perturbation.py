#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import os

from perturb_helper_funcs import *


# Simulation period and number of ensembles
years = list(range(2009, 2019))
num_ensemble = 150

# Initialize lists to hold all perturbed values
all_perturbed_values = {var: [] for var in ["PRECTmms", "FSDS", "FLDS", "TBOT", "WIND", "RH"]}

def perturb_nc_file(year, month, iensemble):
    # standard deviation and mean of perturbations
    sd   = [ln_to_n(1.0, 0.5)[1], ln_to_n(1.0, 0.3)[1], 20.0, 1.0, ln_to_n(1.0, 0.3)[1], 2.0]
    mean = [ln_to_n(1.0, 0.5)[0], ln_to_n(1.0, 0.3)[0],  0.0, 0.0, ln_to_n(1.0, 0.3)[0], 0.0]

    correl = np.array([[ 1.00,  -0.27,  0.26,  0.00,  0.20,  0.31],
                       [-0.27,   1.00,  0.20,  0.65, -0.20, -0.68],
                       [ 0.26,   0.20,  1.00,  0.76,  0.00,  0.00],
                       [ 0.00,   0.65,  0.76,  1.00,  0.00, -0.44],
                       [ 0.20,  -0.20,  0.00,  0.00,  1.00,  0.00],
                       [ 0.31,  -0.68,  0.00, -0.44,  0.00,  1.00]])
   
    fname = ("/home/fernand/JURECA/CLM5_DATA/inputdata/atm/datm7/trial_12345/DE-RuW/local/" +
            str(year) + "-" + str(month).zfill(2) + ".nc")
    outname = ("/home/fernand/JURECA/CLM5_DATA/inputdata/atm/datm7/trial_12345/DE-RuW/local/" +
               "Ensemble/real_" + 
               str(iensemble + 1).zfill(5) + "/" +
               str(year) + "-" + str(month).zfill(2) + ".nc")
               
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    with nc.Dataset(fname) as src, nc.Dataset(outname, "w") as dst:

        copy_attr_dim(src, dst)

        dim_time = src.dimensions["time"].size
        dim_lat  = src.dimensions["lat"].size
        dim_lon  = src.dimensions["lon"].size

        rnd  = np.random.multivariate_normal(np.zeros_like(mean), correl, dim_time*dim_lat*dim_lon)
        
        perturbations = np.zeros_like(rnd)
        perturbations[:, 0] = np.exp(mean[0] + sd[0] * rnd[:, 0])
        perturbations[:, 1] = np.exp(mean[1] + sd[1] * rnd[:, 1])
        perturbations[:, 2] = mean[2] + rnd[:, 2] * sd[2]
        perturbations[:, 3] = mean[3] + rnd[:, 3] * sd[3]
        perturbations[:, 4] = np.exp(mean[4] + sd[4] * rnd[:, 4])
        perturbations[:, 5] = mean[5] + rnd[:, 5] * sd[5]

        mean_precip    = np.mean(perturbations[:, 0])
        mean_shortwave = np.mean(perturbations[:, 1])
        mean_wind      = np.mean(perturbations[:, 4])

        perturbations[:, 0] = np.divide(perturbations[:, 0], mean_precip)
        perturbations[:, 1] = np.divide(perturbations[:, 1], mean_shortwave)
        perturbations[:, 4] = np.divide(perturbations[:, 4], mean_wind)

        # Prepare to store perturbed values
        perturbed_values = {}

        # After generating all random variables
        # save state of random number generator to file
        if not force_seed:
            rnd_state_serialize('forcing')

        # netCDF variables
        # Copy non-perturbed variables:
        for name, var in src.variables.items():
            if name not in ["PRECTmms", "FSDS", "FLDS", "TBOT", "WIND", "RH"]:
                nvar = dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]

        prectmms = dst.createVariable("PRECTmms", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        prectmms.setncatts({"units": u"mm/s", "missing_value": -9.e+33})
        perturbed_prectmms = src.variables["PRECTmms"][:] * perturbations[:, 0].reshape(src.variables["PRECTmms"][:, :, :].shape)
        dst.variables["PRECTmms"][:] = perturbed_prectmms
        perturbed_values["PRECTmms"] = perturbed_prectmms

        fsds = dst.createVariable("FSDS", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        fsds.setncatts({"units": u"W/m^2", "missing_value": -9.e+33})
        perturbed_fsds = src.variables["FSDS"][:] * perturbations[:, 1].reshape(src.variables["FSDS"][:, :, :].shape)
        dst.variables["FSDS"][:] = perturbed_fsds
        perturbed_values["FSDS"] = perturbed_fsds
        
        flds = dst.createVariable("FLDS", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        flds.setncatts({"units": u"W/m^2", "missing_value": -9.e+33})
        perturbed_flds = clm3_5_flds_calc(src.variables["PSRF"][:],
                                          src.variables["RH"][:],
                                          src.variables["TBOT"][:]) + \
                                          perturbations[:, 2].reshape(
                                          src.variables["FSDS"][:, :, :].shape)
        dst.variables["FLDS"][:] = perturbed_flds
        perturbed_values["FLDS"] = perturbed_flds

        tbot = dst.createVariable("TBOT", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=9.96921e+36)
        tbot.setncatts({"height": u"2", "units": u"K", "missing_value": -9.e+33})
        perturbed_tbot = src.variables["TBOT"][:, :, :] + \
                   perturbations[:, 3].reshape(src.variables["TBOT"][:, :, :].shape)
        dst.variables["TBOT"][:, :, :] = perturbed_tbot
        perturbed_values["TBOT"] = perturbed_tbot

        wind = dst.createVariable("WIND", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        wind.setncatts({"units": u"mm/s", "missing_value": -9.e+33})
        perturbed_wind = src.variables["WIND"][:] * perturbations[:, 4].reshape(src.variables["WIND"][:, :, :].shape)
        dst.variables["WIND"][:] = perturbed_wind
        perturbed_values["WIND"] = perturbed_wind
        
        rh = dst.createVariable("RH", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        rh.setncatts({"units": u"%", "missing_value": -9.e+33})
        perturbed_rh = np.clip(src.variables["RH"][:] + 
                               perturbations[:, 5].reshape(
                               src.variables["RH"][:, :, :].shape), 0, 100)  # Clip values between 0 and 100
        dst.variables["RH"][:] = perturbed_rh
        perturbed_values["RH"] = perturbed_rh

        # Store perturbed values for each variable
        for var, values in perturbed_values.items():
            all_perturbed_values[var].append(values.flatten())

# Settings / parameters
rnd_state_file = "forcing_rnd_state.json"
force_seed = False 
if not os.path.isfile(rnd_state_file) or force_seed:
    np.random.seed(12345)
else:
    rnd_state_deserialize('forcing')

for ens in range(num_ensemble):
    for y in years:
        for m in range(1, 13):
            perturb_nc_file(y, m, ens)
        print("Done with year " + str(y) + " ensemble " + str(ens))

# Concatenate all perturbed values for each variable
for var in all_perturbed_values:
    all_perturbed_values[var] = np.concatenate(all_perturbed_values[var])

# Plot histograms
variables = ["Precipitation(mm/s)", "ShortWave(W/m^2)", "LongWave(W/m^2)", "Temperature(K)", "Wind(m/s)", "RH(%)"]
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (var, ax) in enumerate(zip(all_perturbed_values.keys(), axes)):
    ax.hist(all_perturbed_values[var], bins=50, edgecolor='black')
    ax.set_title(variables[i])
    ax.set_xlabel('Perturbed Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
