#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size' : 30})

from helper_funcs import *

# Set plot style
# plt.style.use('ggplot')


def load_data():
    # Read monthly obs data
    filepath_obs = "/p/scratch/cjicg41/eloundou1/Observations/ICOS_data/all_obs_data.csv"
    obs_data = pd.read_csv(filepath_obs, parse_dates=['time'])
    obs_data['time'] = pd.to_datetime(obs_data['time'], format='%Y%m')

    # Directories of ensemble realizations for all sites
    be_bra_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/BE-Bra/"
    cz_bk1_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/CZ-BK1/"
    de_hai_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/DE-Hai/"
    de_hoh_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/DE-HoH/"
    de_obe_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/DE-Obe/"
    de_ruw_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/DE-RuW/"
    dk_vng_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/DK-Vng/"
    es_cnd_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/ES-Cnd/"
    fi_hyy_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/FI-Hyy/"
    fi_sod_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/FI-Sod/"
    fr_pue_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/FR-Pue/"
    it_lav_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/IT-Lav/"
    nl_loo_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/NL-Loo/"
    se_svb_path = "/p/scratch/cjicg41/eloundou1/CLM5_DATA/Archive/lnd/hist/Ensemble/soil/SE-Svb/"

    # Interested variables and ensemble data
    # vars = ['QFLX_EVAP_TOT', 'GPP', 'NEE', 'FSH','TLAI']
    var_sim = ['QFLX_EVAP_TOT']
    be_bra_ds = load_ens_sim(be_bra_path, var_sim)
    cz_bk1_ds = load_ens_sim(cz_bk1_path, var_sim)
    de_hai_ds = load_ens_sim(de_hai_path, var_sim)
    de_hoh_ds = load_ens_sim(de_hoh_path, var_sim)
    de_obe_ds = load_ens_sim(de_obe_path, var_sim)
    de_ruw_ds = load_ens_sim(de_ruw_path, var_sim)
    dk_vng_ds = load_ens_sim(dk_vng_path, var_sim)
    es_cnd_ds = load_ens_sim(es_cnd_path, var_sim)
    fi_hyy_ds = load_ens_sim(fi_hyy_path, var_sim)
    fi_sod_ds = load_ens_sim(fi_sod_path, var_sim)
    fr_pue_ds = load_ens_sim(fr_pue_path, var_sim)
    it_lav_ds = load_ens_sim(it_lav_path, var_sim)
    nl_loo_ds = load_ens_sim(nl_loo_path, var_sim)
    se_svb_ds = load_ens_sim(se_svb_path, var_sim)

    be_bra_df = convert_ens_to_dataframe(be_bra_ds, var_sim)
    cz_bk1_df = convert_ens_to_dataframe(cz_bk1_ds, var_sim)
    de_hai_df = convert_ens_to_dataframe(de_hai_ds, var_sim)
    de_hoh_df = convert_ens_to_dataframe(de_hoh_ds, var_sim)
    de_obe_df = convert_ens_to_dataframe(de_obe_ds, var_sim)
    de_ruw_df = convert_ens_to_dataframe(de_ruw_ds, var_sim)
    dk_vng_df = convert_ens_to_dataframe(dk_vng_ds, var_sim)
    es_cnd_df = convert_ens_to_dataframe(es_cnd_ds, var_sim)
    fi_hyy_df = convert_ens_to_dataframe(fi_hyy_ds, var_sim)
    fi_sod_df = convert_ens_to_dataframe(fi_sod_ds, var_sim)
    fr_pue_df = convert_ens_to_dataframe(fr_pue_ds, var_sim)
    it_lav_df = convert_ens_to_dataframe(it_lav_ds, var_sim)
    nl_loo_df = convert_ens_to_dataframe(nl_loo_ds, var_sim)
    se_svb_df = convert_ens_to_dataframe(se_svb_ds, var_sim)
    

    return obs_data, be_bra_df, cz_bk1_df, de_hai_df, de_hoh_df, de_obe_df, de_ruw_df, dk_vng_df, es_cnd_df, fi_hyy_df, fi_sod_df, fr_pue_df, it_lav_df, nl_loo_df, se_svb_df

def convert_to_per_day(df, site, var, sim_var):
    rename_mapping = {f'{sim_var}': f'{site}_{var}'}
    df[[f'{sim_var}']] = df[[f'{sim_var}']].mul(86400)
    df = df.rename(columns=rename_mapping)
    return df

def process_site_realizations(site_dfs, sites, var, sim_var):
    site_realizations = {}
    for site, site_df in zip(sites, site_dfs):
        ens_df = {realization: convert_to_per_day(df, site, var, sim_var) for realization, df in site_df.items()}
        realizations = {data: timeseries_custom_resample(df, 'datetime', 'MS') for data, df in ens_df.items()}
        site_realizations[site] = realizations
    return site_realizations


def main():
    obs_data, be_bra_df, cz_bk1_df, de_hai_df,de_hoh_df, de_obe_df, de_ruw_df, dk_vng_df, es_cnd_df, fi_hyy_df,fi_sod_df, fr_pue_df, it_lav_df,nl_loo_df, se_svb_df = load_data()
    
    # For reference
    #sim_vars       = ['QFLX_EVAP_TOT', 'GPP', 'NEE', 'FSH', 'TLAI']
    #obs_var_suffix = ['ET', 'GPP_NT_VUT_REF','NEE_VUT_REF','H_F_MDS','TLAI']
    #labels         = ['Evapotranspiration', 'GPP', 'NEE', 'Sensible Heat Flux', 'LAI']
    #units          = [" (mm d$^{-1}$)", " (gC/m$^{2}$/day)", " (gC/m$^{2}$/day)", " (Wm$^{-2}$)", " (gC/m$^{2}$/day)", " (m$^{2}$ m$^{-2}$)"]
    #dir            = ['perturbed_vegetation', 'perturbed_soil', 'perturbed_forcings', 'combined_all']

    sim_var        = 'QFLX_EVAP_TOT'
    var            = 'ET'     
    var_title      = 'Evapotranspiration'
    label          = 'Evapotranspiration'
    unit           = " (mm d$^{-1}$)"
    dir            = 'perturbed_soil'
    pert_factor    = 'Perturbed soil parameters'
    folder         = 'ET'

    # Observed data
    observed_data = {
        'datetime': obs_data['time'], 
        'BE-Bra_'f'{var}': obs_data['BE-Bra_'f'{var}'],  
        'CZ-BK1_'f'{var}': obs_data['CZ-BK1_'f'{var}'],
        'DE-Hai_'f'{var}': obs_data['DE-Hai_'f'{var}'],
        'DE-HoH_'f'{var}': obs_data['DE-HoH_'f'{var}'],
        'DE-Obe_'f'{var}': obs_data['DE-Obe_'f'{var}'],
        'DE-RuW_'f'{var}': obs_data['DE-RuW_'f'{var}'],
        'DK-Vng_'f'{var}': obs_data['DK-Vng_'f'{var}'],
        'ES-Cnd_'f'{var}': obs_data['ES-Cnd_'f'{var}'],
        'FI-Hyy_'f'{var}': obs_data['FI-Hyy_'f'{var}'],
        'FI-Sod_'f'{var}': obs_data['FI-Sod_'f'{var}'],
        'FR-Pue_'f'{var}': obs_data['FR-Pue_'f'{var}'],
        'IT-Lav_'f'{var}': obs_data['IT-Lav_'f'{var}'],
        'NL-Loo_'f'{var}': obs_data['NL-Loo_'f'{var}'],
        'SE-Svb_'f'{var}': obs_data['SE-Svb_'f'{var}']
    }

    sites          = ['BE-Bra', 'CZ-BK1', 'DE-Hai', 'DE-HoH', 'DE-Obe', 'DE-RuW', 'DK-Vng', 
                      'ES-Cnd', 'FI-Hyy', 'FI-Sod', 'FR-Pue', 'IT-Lav', 'NL-Loo', 'SE-Svb']
    
    site_full      = ['BE-Brasschaat', 'CZ-Bily Kriz', 'DE-Hainich','DE-Hohes Holz', 'DE-Oberbärenburg', 'DE-Wüstebach', 'DK-Voulundgard', 
                     'ES-Conde', 'FI-Hyytiälä','FI-Sodankylä', 'FR-Puechabon', 'IT-Lavarone', 'NL-Loobos', 'SE-Svartberget']
    
    site_dfs       = [be_bra_df, cz_bk1_df, de_hai_df,de_hoh_df, de_obe_df, de_ruw_df, dk_vng_df, 
                      es_cnd_df, fi_hyy_df,fi_sod_df, fr_pue_df, it_lav_df,nl_loo_df, se_svb_df]
    

    site_realizations = process_site_realizations(site_dfs, sites, var, sim_var)

    variance_file = variance_by_site(site_realizations, sites, var)

    result_df = calculate_lower_and_upper_bound_values(site_realizations, sites, var)
    
    # Plot the enveloping of observed variables by ensemble spread
    observed_df = pd.DataFrame(observed_data)
    merged_df = result_df.merge(observed_df, on='datetime')

    # Initialize a dictionary to store the observed time ranges for each site
    observed_time_ranges = {}
    for site in sites:
        observed_time_ranges[site] = observed_df[observed_df[f'{site}_{var}'].notnull()]['datetime']

    save_dir = f"/p/scratch/cjicg41/eloundou1/codes/ensemble_plots/{dir}/{folder}"
    save_performance_dir = f"/p/scratch/cjicg41/eloundou1/codes/ensemble_plots/{dir}/{folder}/performance"
    plot_ensemble_spread(site_realizations, observed_df, merged_df, observed_time_ranges, pert_factor, var, label, unit, save_dir, sites, site_full, var_title)
    plot_ensemble_spread_mean(site_realizations, observed_df, merged_df, observed_time_ranges, pert_factor, var, label, unit, save_dir, sites, site_full, var_title, save_performance_dir)

    # Plot percent coverage by ensemble spread and save data to csv
    save_cover = f"/p/scratch/cjicg41/eloundou1/codes/ensemble_plots/{dir}/{folder}"
    plot_coverage(merged_df, observed_time_ranges, var, var_title, sites, save_cover)

    # Plot all ensemble realizations against observation
    plot_ensemble_realizations(site_realizations, obs_data, var, label, unit, save_dir, sites, site_full)
    
    # Monthly ensemble spread and metrics
    site_ensemble_means = {}
    for site, realization_data in site_realizations.items():
        site_df = pd.concat(realization_data.values())
        site_ensemble_mean = site_df.groupby('datetime').mean().reset_index()
        site_ensemble_means[site] = site_ensemble_mean

    # Observed data
    data_obs = {
        'datetime'  : obs_data['time'], 
        'BE-Bra_obs': obs_data['BE-Bra_'f'{var}'],  
        'CZ-BK1_obs': obs_data['CZ-BK1_'f'{var}'],
        'DE-Hai_obs': obs_data['DE-Hai_'f'{var}'],
        'DE-HoH_obs': obs_data['DE-HoH_'f'{var}'],
        'DE-Obe_obs': obs_data['DE-Obe_'f'{var}'],
        'DE-RuW_obs': obs_data['DE-RuW_'f'{var}'],
        'DK-Vng_obs': obs_data['DK-Vng_'f'{var}'],
        'ES-Cnd_obs': obs_data['ES-Cnd_'f'{var}'],
        'FI-Hyy_obs': obs_data['FI-Hyy_'f'{var}'],
        'FI-Sod_obs': obs_data['FI-Sod_'f'{var}'],
        'FR-Pue_obs': obs_data['FR-Pue_'f'{var}'],
        'IT-Lav_obs': obs_data['IT-Lav_'f'{var}'],
        'NL-Loo_obs': obs_data['NL-Loo_'f'{var}'],
        'SE-Svb_obs': obs_data['SE-Svb_'f'{var}']
    }

    data_ens = {
        'datetime'  : site_ensemble_means['BE-Bra']['datetime'], 
        'BE-Bra_ens': site_ensemble_means['BE-Bra']['BE-Bra_'f'{var}'],  
        'CZ-BK1_ens': site_ensemble_means['CZ-BK1']['CZ-BK1_'f'{var}'],
        'DE-Hai_ens': site_ensemble_means['DE-Hai']['DE-Hai_'f'{var}'],
        'DE-HoH_ens': site_ensemble_means['DE-HoH']['DE-HoH_'f'{var}'],
        'DE-Obe_ens': site_ensemble_means['DE-Obe']['DE-Obe_'f'{var}'],
        'DE-RuW_ens': site_ensemble_means['DE-RuW']['DE-RuW_'f'{var}'],
        'DK-Vng_ens': site_ensemble_means['DK-Vng']['DK-Vng_'f'{var}'],
        'ES-Cnd_ens': site_ensemble_means['ES-Cnd']['ES-Cnd_'f'{var}'],
        'FI-Hyy_ens': site_ensemble_means['FI-Hyy']['FI-Hyy_'f'{var}'],
        'FI-Sod_ens': site_ensemble_means['FI-Sod']['FI-Sod_'f'{var}'],
        'FR-Pue_ens': site_ensemble_means['FR-Pue']['FR-Pue_'f'{var}'],
        'IT-Lav_ens': site_ensemble_means['IT-Lav']['IT-Lav_'f'{var}'],
        'NL-Loo_ens': site_ensemble_means['NL-Loo']['NL-Loo_'f'{var}'],
        'SE-Svb_ens': site_ensemble_means['SE-Svb']['SE-Svb_'f'{var}']
    }

    obs_df = pd.DataFrame(data_obs)
    ens_df = pd.DataFrame(data_ens)
    obs_ens_merged = ens_df.merge(obs_df, on='datetime')
    
    obs_time_ranges = {}
    for site in sites:
        obs_time_ranges[site] = obs_df[obs_df[f'{site}_obs'].notnull()]['datetime']

    save_month_dir   = f"/p/scratch/cjicg41/eloundou1/codes/ensemble_plots/{dir}/{folder}"
    save_metrics_dir = f"/p/scratch/cjicg41/eloundou1/codes/ensemble_plots/{dir}/{folder}/metrics"
    plot_monthly_line_plots(obs_ens_merged, result_df, label, unit, save_month_dir, sites, site_full, obs_time_ranges, save_metrics_dir, var_title)


if __name__ == "__main__":
    main()

