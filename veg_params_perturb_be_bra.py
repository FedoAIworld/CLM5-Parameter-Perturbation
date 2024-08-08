#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
from pyDOE import lhs
from perturb_helper_funcs import *
import matplotlib.pyplot as plt


def generate_vname(ensemble_number):
    base_path = "/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/paramdata/ensemble/trial_54321/BE-Bra/"
    filename = f"clm5_params.c171117.nc_{str(ensemble_number + 1).zfill(5)}.nc"
    return os.path.join(base_path, filename)

def load_parameter_values(ensemble_number, parameter_name):
    vname = generate_vname(ensemble_number)
    with nc.Dataset(vname) as dataset:
        if parameter_name in dataset.variables:
            parameter_values = dataset[parameter_name][:]
            return parameter_values.flatten()
        else:
            print(f"Warning: {parameter_name} not found in {vname}")
            return None

def perturb_value(param_name, param_min, param_max):
    if param_name in ['medlynintercept', 'tpu25ratio', 'tpuha', 'vcmaxha']:
        # For log-transformed parameters, sample in log space
        log_param_min = np.log10(param_min)
        log_param_max = np.log10(param_max)
        log_lhs_samples = lhs(1, samples=1)
        perturbation_log = log_param_min + log_lhs_samples[0, 0] * (log_param_max - log_param_min)
        perturbation = np.power(10, perturbation_log)  # Back-transform to original space
    else:
        # For non-log-transformed parameters, sample in the specified range
        samples = lhs(1, samples=1)
        perturbation = param_min + samples[0, 0] * (param_max - param_min)
    # Ensure perturbation is within the specified range
    perturbation = np.clip(perturbation, param_min, param_max)
    return perturbation

def perturb_vegetation_parameters(iensemble, parameter_ranges, veg_type_positions):
    vorig = "/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/paramdata/ensemble/trial_54321/BE-Bra/clm5_params.c171117.nc"
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
            nvar = dst.createVariable(name, var.datatype, var.dimensions)
            dst[name].setncatts(src[name].__dict__)
            dst[name][:] = src[name][:]
        
        # Perturb parameters in the specified ranges for specified vegetation types
        for param_name, (param_min, param_max) in parameter_ranges.items():
            for veg_type, position in veg_type_positions.items():
                perturbation = perturb_value(param_name, param_min, param_max)
                
                if param_name not in dst.variables:
                    continue

                param_var_shape = dst[param_name].shape
                print(f"Perturbing {param_name} at position {position} (Dimensions: {param_var_shape})")

                if len(param_var_shape) == 0:  # Scalar parameter
                    dst[param_name][:] = perturbation
                elif len(param_var_shape) == 1:  # Vector parameter
                    if position < param_var_shape[0]:
                        dst[param_name][position] = perturbation
                    else:
                        print(f"Warning: Position {position} is out of bounds for parameter {param_name}")
                elif len(param_var_shape) == 2:  # Matrix parameter
                    if position < param_var_shape[1]:
                        for segment in range(param_var_shape[0]):
                            dst[param_name][segment, position] = perturbation
                    else:
                        print(f"Warning: Position {position} is out of bounds for parameter {param_name}")

def plot_perturbed_parameters(parameter_names, num_ensemble, output_dir):
    default_values = {param_name: [] for param_name in parameter_names}
    perturbed_values = {param_name: [] for param_name in parameter_names}

    for ens in range(num_ensemble):
        for param_name in parameter_names:
            default_value = load_parameter_values(ens, param_name)
            if default_value is not None:
                default_values[param_name].extend(default_value)
                perturbed_values[param_name].append(np.mean(default_value))  # Assuming mean perturbation

    for param_name in parameter_names:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(default_values[param_name], bins=20, color='blue', edgecolor='black', alpha=0.7, label='Default')
        ax.hist(perturbed_values[param_name], bins=20, color='red', edgecolor='black', alpha=0.7, label='Perturbed')
        ax.set_title(param_name)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        output_file = os.path.join(output_dir, f"{param_name}_histogram.png")
        plt.savefig(output_file)
        plt.close(fig)

def main():
    num_ensemble = 130
    random_seed = 54321

    # Define the parameters and their perturbation ranges for ENF and DBF
    enf_parameter_ranges = {
        'jmaxb0':          (0.01, 0.05),
        'jmaxb1':          (0.05, 0.25),
        'slatop':          (0.0073, 0.0127),
        'leafcn':          (40.6, 75.4),
        'medlynintercept': (1, 200000),
        'medlynslope':     (1.29, 4.7),
        'theta_cj':        (0.8, 0.99),
        'tpu25ratio':      (0.0835, 0.501),
        'wc2wjb0':         (0.5, 1.5),
        'jmaxha':          (34000, 78000),
        'lmrha':           (23195, 69585),
        'vcmaxha':         (45000, 173000),
        'tpuha':           (45000, 173000),
        'leaf_long':       (2.65, 3.97),
        'kmax':            (3.00E-09, 3.80E-08),
        'rholnir':         (0.31, 0.51)
    }

    dbf_parameter_ranges = {
        'jmaxb0':          (0.01, 0.05),
        'jmaxb1':          (0.05, 0.25),
        'leafcn':          (16.45, 30.55),
        'medlynintercept': (1, 200000),
        'medlynslope':     (3.19, 5.11),
        'theta_cj':        (0.8, 0.99),
        'wc2wjb0':         (0.5, 1.5),
        'stem_leaf':       (1.84, 2.76),
        'FUN_fracfixers':  (0, 1),
        'kmax':            (1.20E-08, 2.20E-07),
        'psi50':           (-351000, -189000),
        'rholnir':         (0.36, 0.48),
        'taulnir':         (0.23, 0.53)
    }

    veg_type_positions = {
        'ENF': 1,
        'DBF': 7
    }

    rnd_state_file = "veg_rnd_state.json"
    if not os.path.isfile(rnd_state_file):
        np.random.seed(random_seed)
    else:
        rnd_state_deserialize('veg')

    parameter_names = list(set(enf_parameter_ranges.keys()).union(set(dbf_parameter_ranges.keys())))

    # Loop through ensemble members and perturb the parameters
    for ens in range(num_ensemble):
        perturb_vegetation_parameters(ens, enf_parameter_ranges, veg_type_positions)
        perturb_vegetation_parameters(ens, dbf_parameter_ranges, veg_type_positions)
        print(f"Ensemble member {ens + 1} perturbed and saved to output file.")

    # After generating all random variables
    # save state of random number generator to file
    rnd_state_serialize('veg')
    
    # Define output directory for plots
    output_dir = "/home/fernand/JURECA/CLM5_DATA/inputdata/lnd/clm2/paramdata/ensemble/trial_54321/BE-Bra/plots/"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the accumulated parameter values
    plot_perturbed_parameters(parameter_names, num_ensemble, output_dir)

if __name__ == "__main__":
    main()
