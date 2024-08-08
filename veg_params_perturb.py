#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
from pyDOE import lhs
from perturb_helper_funcs import *


def generate_vname(ensemble_number):
    base_path = "/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/paramdata/ensemble/trial_54321/DE-RuW"
    filename = f"clm5_params.c171117.nc_{str(ensemble_number + 1).zfill(5)}.nc"
    return os.path.join(base_path, filename)

def load_parameter_values(ensemble_number, parameter_name):
    vname = generate_vname(ensemble_number)
    with nc.Dataset(vname) as dataset:
        parameter_values = dataset[parameter_name][:]
    return parameter_values.flatten()

def perturb_vegetation_parameters(iensemble, parameter_ranges):
    vorig = "/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/paramdata/ensemble/trial_54321/DE-RuW/clm5_params.c171117.nc"
    vname = generate_vname(iensemble)

    
    with nc.Dataset(vorig) as src, nc.Dataset(vname, "w") as dst:
        # Copy attributes
        copy_attr_dim(src, dst)

        # Check if dimensions already exist, and create them if not
        for name, dim in src.dimensions.items():
            if name not in dst.dimensions:
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
        
        # Copy non-perturbed variables
        for name, var in src.variables.items():
            if name not in parameter_ranges:
                nvar = dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]
        
        # Perturb parameters in the specified ranges
        for param_name, (param_min, param_max) in parameter_ranges.items():
            if param_name in ['medlynintercept', 'tpu25ratio', 'fff', 'tpuha', 'vcmaxha', 'lmrhd']:
                # For log-transformed parameters, sample in log space
                log_param_min    = np.log10(param_min)
                log_param_max    = np.log10(param_max)
                log_lhs_samples  = lhs(1, samples=1)
                perturbation_log = log_param_min + log_lhs_samples[0, 0] * (log_param_max - log_param_min)
                perturbation     = np.power(10, perturbation_log)  # Back-transform to original space
            else:
                # For non-log-transformed parameters, sample in the specified range
                samples      = lhs(1, samples=1)
                perturbation = param_min + samples[0, 0] * (param_max - param_min)

            # Create the variable in the NetCDF file
            param_var = dst.createVariable(param_name, 'f8', src.variables[param_name].dimensions)
            param_var[:] = perturbation


def main():
    num_ensemble = 150
    random_seed = 54321

    # Define the parameters and their perturbation ranges
    parameter_ranges = {
        'jmaxb0':          (0.01, 0.05),
        'jmaxb1':          (0.05, 0.25),
        'slatop':          (0.0073, 0.0127),
        'leafcn':          (40.6, 75.4),
        'medlynintercept': (1, 200000),
        'medlynslope':     (1.29, 4.7),
        'theta_cj':        (0.8, 0.99),
        'tpu25ratio':      (0.0835, 0.501),
        'wc2wjb0':         (0.5, 1.5),
        #'lmr_intercept_atkin': (1.77, 2.64),
        'jmaxha':          (34000, 78000),
        'lmrha':           (23195, 69585),
        #'lmrhd':           (75325, 225975),
        'vcmaxha':         (45000, 173000),
        'tpuha':           (45000, 173000),
        #'stem_leaf':       (1.2, 1.8),
        #'froot_leaf':      (1.2, 1.8),
        'leaf_long':       (2.64, 3.96),
        #'FUN_fracfixers':  (0, 1),
        #'krmax':           (5.40E-11, 5.10E-10),
        'kmax':            (3.00E-09, 3.80E-08),
        #'fff':             (0.02, 5),
        #'psi50':           (-351000, -189000),
        #'rhosnir':         (0.424, 0.636),
        'rholnir':         (0.31, 0.51),
        #'taulnir':         (0.23, 0.53)
    }
    
    rnd_state_file = "veg_rnd_state.json"
    if not os.path.isfile(rnd_state_file):
        np.random.seed(random_seed)
    else:
        rnd_state_deserialize('veg')
    
    # Loop through ensemble members and perturb the parameters
    for ens in range(num_ensemble):
        perturb_vegetation_parameters(ens, parameter_ranges)
        print(f"Ensemble member {ens + 1} perturbed and saved to output file.")
    
    # After generating all random variables
    # save state of random number generator to file
    rnd_state_serialize('veg')

    ## Define the parameters for visualization
    #parameters_to_visualize = ['medlynintercept', 'medlynslope', 'jmaxb1', 'vcmaxha', 'theta_cj']
    #
    ## Accumulate parameter values across all ensemble members
    #all_parameter_values = {param_name: [] for param_name in parameters_to_visualize}
    #for ens in range(num_ensemble):
    #    for param_name in parameters_to_visualize:
    #        parameter_values = load_parameter_values(ens, param_name)
    #        all_parameter_values[param_name].extend(parameter_values)
#
    ## Plot the accumulated parameter values
    #fig, axes = plt.subplots(nrows=1, ncols=len(parameters_to_visualize), figsize=(15, 5))
    #for i, param_name in enumerate(parameters_to_visualize):
    #    parameter_values = all_parameter_values[param_name]
    #    axes[i].hist(parameter_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    #    axes[i].set_title(param_name)
    #    axes[i].set_xlabel('Parameter Value')
    #    axes[i].set_ylabel('Frequency')
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()
