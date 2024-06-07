#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import xarray as xr
import numpy as np
import pandas as pd

import calendar
from functools import reduce
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns



# preprocess variables of interest
def preprocess(ds, clm_vars):
    '''
    Preprocess the variables interested.
    ds:       netcdf dataset
    clm_vars: simulation variables interested 
    '''
    return ds[clm_vars]

# load ens data
def load_ens_sim(directory, clm_vars):
    sim_ds = {}
    file_list = os.listdir(directory)
    file_list.sort()  # Sort the file list in ascending order

    for filename in file_list:
        if filename.endswith('.nc'):
            file_path = os.path.join(directory, filename)
            parts = filename.split('.')
            ensemble_member = int(parts[1].split('_')[1])

            # Extract the year using regular expression
            match = re.search(r'\d{4}', filename)
            if match:
                year = int(match.group())
            else:
                continue

            clm_ds = xr.open_dataset(file_path, decode_times=True)
            clm_ds = preprocess(clm_ds, clm_vars)

            # Append the dataset to the corresponding ensemble member key
            if f'real_{ensemble_member}' not in sim_ds:
                sim_ds[f'real_{ensemble_member}'] = clm_ds
            else:
                sim_ds[f'real_{ensemble_member}'] = xr.concat([sim_ds[f'real_{ensemble_member}'], clm_ds], dim='time')

    return sim_ds

# function to resample
def timeseries_custom_resample(df, datetime, freq):
    """ The function resamples timeseries data at a defined resolution."""
    df = df.set_index(str(datetime))
    df = df.resample(str(freq)).mean()
    df = df.reset_index()
    return df

def convert_ens_to_dataframe(dataset, var_list):
    '''
    Convert dataset to dictionary of dataframes for analysis on pandas.
    dataset:  the netcdf dataset
    var_list: the variables of interest
    '''
    dfs = {}  # Dictionary to store DataFrames for each realization

    # extract variables from dataset
    for key, value in dataset.items():
        if key.startswith('real_'):
            realization = int(key.split('_')[1])
            time = value['time'].values.astype('datetime64[ns]')
            df = pd.DataFrame({'datetime': time})
            for var in var_list:
                field = np.ravel(value[var])
                df[var] = field
            dfs[realization] = df
    
    return dfs

def calculate_performance_metrics(observed, predicted):
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    ubrmse = rmse / np.mean(observed)
    r_value = r2_score(observed, predicted)
    bias = np.mean(predicted - observed)
    pbias = 100 * bias / np.mean(observed)
    return rmse, ubrmse, r_value, bias, pbias

def calculate_lower_and_upper_bound_values(site_realizations, sites, var):
    # Initialize dictionaries to store var values for each site and time
    var_values_by_site_time = {site: {} for site in sites}

    # Iterate through each site's realizations in the dictionary
    for site, site_realization_data in site_realizations.items():
        for realization, df in site_realization_data.items():
            for index, row in df.iterrows():
                datetime = index  # Use the index as the datetime value
                var_value = row[f'{site}_{var}']
                
                # If datetime not in the dictionary, create an entry
                if datetime not in var_values_by_site_time[site]:
                    var_values_by_site_time[site][datetime] = [var_value]
                else:
                    var_values_by_site_time[site][datetime].append(var_value)

    # Initialize dictionaries to store min and max values for var at each time step for each site
    lb_ub_var_values_by_site_time = {site: {'lb': {}, 'ub': {}} for site in sites}

    # Calculate the min and max values of var at each time step for each site
    for site in sites:
        for datetime, values in var_values_by_site_time[site].items():
            # Calculate 99% confidence intervals
            lower_bound, upper_bound = np.percentile(values, [0.5, 99.5])
            # Store the values in the dictionary
            lb_ub_var_values_by_site_time[site]['lb'][datetime] = lower_bound
            lb_ub_var_values_by_site_time[site]['ub'][datetime] = upper_bound
            

    # Determine the maximum length of lower and upper bound arrays across all sites
    max_length = max(len(lb_ub_var_values_by_site_time[site]['lb']) for site in sites)

    # Pad arrays with np.nan for sites with shorter arrays
    for site in sites:
        lb_values = list(lb_ub_var_values_by_site_time[site]['lb'].values())
        ub_values = list(lb_ub_var_values_by_site_time[site]['ub'].values())

        if len(lb_values) < max_length:
            lb_values += [np.nan] * (max_length - len(lb_values))
        if len(ub_values) < max_length:
            ub_values += [np.nan] * (max_length - len(ub_values))

        # Update dictionaries with padded arrays
        lb_ub_var_values_by_site_time[site]['lb'] = lb_values
        lb_ub_var_values_by_site_time[site]['ub'] = ub_values

    # Create a datetime64 column from '2009-01-01' to '2018-12-31'
    datetime_column = pd.date_range(start='2009-01-01', end='2018-12-31', freq='MS')

    # Create a DataFrame with min and max values of var for each site
    result_data = {'datetime': datetime_column}
    for site in sites:
        result_data[f'{site}_lb'] = lb_ub_var_values_by_site_time[site]['lb']
        result_data[f'{site}_ub'] = lb_ub_var_values_by_site_time[site]['ub']

    result_df = pd.DataFrame(result_data)
    return result_df

def plot_ensemble_spread(site_realizations, observed_df, merged_df, observed_time_ranges, pert_factor, var, label, unit, save_dir, sites, site_full, var_title):
    os.makedirs(save_dir, exist_ok=True)
    
    for site, site_full in zip(sites, site_full):
        #plt.rcParams.update(font)
        plt.figure(figsize=(15, 12))

        # Filter the result_df based on the observed time range for the current site
        site_result_df = merged_df[merged_df['datetime'].isin(observed_time_ranges[site])]

        plt.fill_between(site_result_df['datetime'], site_result_df[f'{site}_lb'], site_result_df[f'{site}_ub'], color='#f03b20', alpha=0.3, label='Ens. spread')
        plt.plot(site_result_df['datetime'], site_result_df[f'{site}_{var}'], marker='o', markersize=5, linestyle='--', label='Observation', color='black')

        plt.xlabel('Date')
        plt.ylabel(f'{label} {unit}')
        plt.title(f'{pert_factor} at {site_full}', y=1.02)
        plt.xticks(rotation=20, ha='center')
        plt.legend()
        plt.grid(True)

        plot_subdir = os.path.join(save_dir, f'{site.lower()}')
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)

        plot_filename = os.path.join(plot_subdir, f'{site.lower()}_{var_title}.png')
        plt.savefig(plot_filename)

        plt.close()

def plot_ensemble_spread_mean(site_realizations, observed_df, merged_df, observed_time_ranges, pert_factor, var, label, unit, save_dir, sites, site_full, var_title, save_performance_dir):
    os.makedirs(save_dir, exist_ok=True)

    performance_metrics = {'Site': [], 'RMSE': [], 'ubRMSE': [], 'r': [], 'Bias': [], 'PBIAS': []}

    for site, site_full in zip(sites, site_full):
        #plt.rcParams.update(font)
        plt.figure(figsize=(15, 12))

        # Calculate ensemble mean for the current site
        ensemble_means = pd.DataFrame()
        for realization, df in site_realizations[site].items():
            ensemble_means[f'{site}_{var}'] = df[f'{site}_{var}']
        ensemble_means['datetime'] = df['datetime']
        ensemble_means = ensemble_means.groupby('datetime').mean().reset_index()

        # Filter the observed data based on the observed time range for the current site
        site_ens_df = ensemble_means[ensemble_means['datetime'].isin(observed_time_ranges[site])]
        ensemble_values = site_ens_df[f'{site}_{var}']

        # Filter the observed data based on the observed time range for the current site
        site_observed_df = observed_df[observed_df['datetime'].isin(observed_time_ranges[site])]
        observed_values = site_observed_df[f'{site}_{var}']

        # Extract the corresponding columns from the merged DataFrame
        site_result_df = merged_df[merged_df['datetime'].isin(observed_time_ranges[site])]
    

        # Calculate performance metrics
        rmse, ubrmse, r_value, bias, pbias = calculate_performance_metrics(observed_values, ensemble_values)
        performance_metrics['Site'].append(site)
        performance_metrics['RMSE'].append(rmse)
        performance_metrics['ubRMSE'].append(ubrmse)
        performance_metrics['r'].append(r_value)
        performance_metrics['Bias'].append(bias)
        performance_metrics['PBIAS'].append(pbias)


        # Plot ensemble spread
        plt.fill_between(site_observed_df['datetime'], site_result_df[f'{site}_lb'], site_result_df[f'{site}_ub'], color='#f03b20', alpha=0.3, label='Ens. spread')
        plt.plot(site_observed_df['datetime'], ensemble_values, marker='o', markersize=5, linestyle='--', label='Ensemble Mean', color='blue')
        plt.plot(site_observed_df['datetime'], observed_values, marker='s', markersize=5, linestyle='-', label='Observation', color='black')

        plt.xlabel('Date')
        plt.ylabel(f'{label} {unit}')
        plt.title(f'{pert_factor} at {site_full}', y=1.02)
        plt.xticks(rotation=20, ha='center')
        plt.legend()
        plt.grid(True)

        plot_subdir = os.path.join(save_dir, f'{site.lower()}')
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)

        plot_filename = os.path.join(plot_subdir, f'{site.lower()}_{var_title}.png')
        plt.savefig(plot_filename)

        plt.close()

    # Save performance metrics as CSV
    performance_metrics_df = pd.DataFrame(performance_metrics)
    performance_metrics_dir = os.path.join(save_performance_dir)
    if not os.path.exists(performance_metrics_dir):
        os.makedirs(performance_metrics_dir)
    
    performance_csv_path = os.path.join(performance_metrics_dir, f'performance_{var_title}.csv')
    performance_metrics_df.to_csv(performance_csv_path, index=False)


def plot_coverage(merged_df, observed_time_ranges, var, var_title, sites, save_cover):
    # Directory to save coverage files
    coverage_dir = os.path.join(save_cover, 'coverage')
    if not os.path.exists(coverage_dir):
        os.makedirs(coverage_dir)

    # Calculate non-NaN counts for each site
    available_data = {}
    for site in sites:
        obs_values = merged_df[f'{site}_{var}']
        valid_indices = merged_df['datetime'].isin(observed_time_ranges[site])
        obs_values = obs_values[valid_indices]
        available_data[site] = np.sum(~np.isnan(obs_values))

    # Calculate percent coverage excluding NaN values
    coverage_percentages = {}
    for site in sites:
        obs_values = merged_df[f'{site}_{var}']
        lb_values = merged_df[f'{site}_lb']
        ub_values = merged_df[f'{site}_ub']

        # Filter ensemble spread data to match the observed data time range
        valid_indices = merged_df['datetime'].isin(observed_time_ranges[site])
        lb_values = lb_values[valid_indices]
        ub_values = ub_values[valid_indices]

        # Align arrays using the same index
        obs_values = obs_values[valid_indices]

        within_range = np.logical_and(obs_values >= lb_values, obs_values <= ub_values)
        coverage_percentages[site] = np.sum(within_range) / available_data[site] * 100

    # Create a DataFrame for coverage percentages
    coverage_df = pd.DataFrame(coverage_percentages.items(), columns=['Site', 'CoveragePercentage'])

    # Add the available obs data to the DataFrame
    coverage_df['Obs_Count'] = coverage_df['Site'].map(available_data)

    # Save the coverage percentages to a CSV file
    coverage_csv_path = os.path.join(coverage_dir, 'coverage.csv')
    coverage_df.to_csv(coverage_csv_path, index=False)

    # Plot the coverage percentages
    # plt.rcParams.update(font)
    # Sort the DataFrame by coverage percentage in descending order
    coverage_df = coverage_df.sort_values(by='CoveragePercentage', ascending=False)
    plt.figure(figsize=(15, 12))

    plt.plot(coverage_df['Site'], coverage_df['CoveragePercentage'], marker='o', markersize=5, linestyle='--', color='red')

    plt.xlabel('Site')
    plt.ylabel("Coverage Percentage (%)")
    plt.title(f'Coverage of Observed {var_title} by Ensemble Spread', y=1.02)
    plt.xticks(rotation=20, ha='right')
    plt.grid(True)
    plt.ylim(0, max(coverage_df['CoveragePercentage']+2))
    plt.tight_layout()

    # Save the plot in the specified directory
    plot_cover = os.path.join(coverage_dir, f'all_coverage_{var_title}.png')
    plt.savefig(plot_cover)
    plt.close()

def plot_ensemble_realizations(site_realizations, obs_data, var, label, unit, save_dir, sites, site_full):
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    for site, site_name in zip(sites, site_full):
        #plt.rcParams.update(font)
        fig, ax = plt.subplots(figsize=(15, 12))
        plt.suptitle(f'Ensemble Realizations for {site_name}')

        site_var_name = f'{site}_{var}'

        # Plot each realization against the observation data using Seaborn lineplot
        for r, df in site_realizations[site].items():
            sns.lineplot(data=df, x='datetime', y=site_var_name, ax=ax, label=f'Realization {r}')
        

        # Plot the observation data using Seaborn lineplot
        obs_line = sns.lineplot(data=obs_data, x='time', y=site_var_name, ax=ax, linestyle='--', color='black', label='Obs')

        ax.set_xlabel('Date')
        ax.set_ylabel(f'{label} {unit}')
        #ax.set_title(f'{label} - {site_full}', fontweight='bold')

        # Create a separate legend for the observation data
        obs_proxy = plt.Line2D([], [], linestyle='--', color='black', label='Obs')
        ax.legend(handles=[obs_proxy], loc='upper right')

        # Save the plot to the specified directory with a descriptive filename
        plot_subdir = os.path.join(save_dir, site.lower())
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)
        plot_filename = os.path.join(plot_subdir, f'{site.lower()}_real.png')
        plt.savefig(plot_filename)
        plt.close()

def plot_monthly_line_plots(obs_ens_merged, result_df, label, unit, save_dir, sites, site_full, obs_time_ranges, save_metrics_dir, var_title):
    """Plot monthly line plots for observations, ensemble mean, and ensemble spread."""
    os.makedirs(save_dir, exist_ok=True)
    metrics_list = []
    
    for site, site_name in zip(sites, site_full):
        observed_range = obs_time_ranges[site]
        site_data = obs_ens_merged[obs_ens_merged['datetime'].isin(observed_range)]
        ens_spread_df_filtered = result_df[result_df['datetime'].isin(observed_range)]
        
        site_data['month'] = site_data['datetime'].dt.month
        ens_spread_df_filtered['month'] = ens_spread_df_filtered['datetime'].dt.month
        
        obs_col = f'{site}_obs'
        ens_col = f'{site}_ens'
        
        site_data = site_data.dropna(subset=[obs_col])
        ensemble_mean_filtered = site_data[[obs_col, ens_col, 'datetime']]
        
        #plt.rcParams.update(font)
        fig, ax = plt.subplots(figsize=(15, 12))
        fig.suptitle(f'Monthly {label} at {site_name}')
        
        obs_monthly = site_data.groupby(site_data['datetime'].dt.month)[obs_col].mean()
        ens_monthly = ensemble_mean_filtered.groupby(ensemble_mean_filtered['datetime'].dt.month)[ens_col].mean()

        obs_monthly_df = pd.DataFrame({'index': obs_monthly.index, 'values': obs_monthly.values})
        ens_monthly_df = pd.DataFrame({'index': ens_monthly.index, 'values': ens_monthly.values})
    
        columns_to_group = [f'{site}_lb', f'{site}_ub']
        monthly_ens_spread = ens_spread_df_filtered.groupby('month')[columns_to_group].mean()
        
        r2 = r2_score(obs_monthly, ens_monthly)
        correlation, _ = pearsonr(obs_monthly, ens_monthly)
        bias = np.mean(ens_monthly - obs_monthly)
        rmse = np.sqrt(np.mean((ens_monthly - obs_monthly) ** 2))

        metrics_list.append({'Site': site_name, 'R-squared': r2, 'Correlation': correlation, 'Bias': bias, 'RMSE': rmse})
        
        obs_line = sns.lineplot(data=obs_monthly_df, x='index', y='values', ax=ax, marker='o', markersize=5,
                                linestyle='--', label='Observations', color='black')
        mean_line = sns.lineplot(data=ens_monthly_df, x='index', y='values', ax=ax, marker='s', markersize=5, color='#f03b20',
                                 linestyle='-', label='Ensemble Mean')
        ax.fill_between(monthly_ens_spread.index, monthly_ens_spread[f'{site}_lb'], monthly_ens_spread[f'{site}_ub'],
                        alpha=0.3, color='#feb24c')

        ax.set_xlabel('Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])

        ax.set_ylabel(f'{label} {unit}')

        ax.legend(loc='best')

        plot_subdir = os.path.join(save_dir, f'{site.lower()}')
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)

        plot_filename = os.path.join(plot_subdir, f'{site.lower()}_metrics.png')
        plt.savefig(plot_filename)

        plt.close()

    metrics_df = pd.DataFrame(metrics_list)
    metrics_dir = os.path.join(save_metrics_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    metrics_csv_path = os.path.join(save_metrics_dir, f'metrics_{var_title}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

def variance_by_site(site_realizations, sites, var):
    # Initialize a dictionary to store all values for each site
    all_values_by_site = {site: [] for site in sites}

    # Iterate through each site's realizations in the dictionary
    for site, site_realization_data in site_realizations.items():
        for realization, df in site_realization_data.items():
            all_values_by_site[site].extend(df[f'{site}_{var}'].values)

    # Calculate the overall sample variance for each site
    variance_by_site = {site: round(np.var(values, ddof=1), 2) for site, values in all_values_by_site.items()}

    # Convert the variance dictionary to a DataFrame
    variance_df = pd.DataFrame(list(variance_by_site.items()), columns=['Site', f'variance_{var}'])

    # Save the variance DataFrame to a CSV file
    csv_filename = os.path.join(f'{var}_variancefile', 'all_sites_variances.csv')
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    variance_df.to_csv(csv_filename, index=False)

    return variance_by_site


def timestep_realizations_save_reals(site_realizations, sites, var):
    # Initialize a dictionary to store var values for each site and time
    var_values_by_site_time = {f'{site}_{var}': {} for site in sites}

    # Iterate through each site's realizations in the dictionary
    for site, site_realization_data in site_realizations.items():
        for realization, df in site_realization_data.items():
            for index, row in df.iterrows():
                datetime = index  # Use the index as the datetime value
                var_value = row[f'{site}_{var}']

                # If datetime not in the dictionary, create an entry
                if datetime not in var_values_by_site_time[f'{site}_{var}']:
                    var_values_by_site_time[f'{site}_{var}'][datetime] = [var_value]
                else:
                    var_values_by_site_time[f'{site}_{var}'][datetime].append(var_value)

    # Create CSV files for each site
    for site_var, site_data in var_values_by_site_time.items():
        # Convert the dictionary of ensemble values to a DataFrame
        site_df = pd.DataFrame(site_data).T.reset_index()
        site_df.columns = ['time'] + [f'real_{i+1}' for i in range(len(site_df.columns)-1)]

        # Save DataFrame to CSV file for this site
        site_name = site_var.split('_')[0]  # Extract site name from the variable name
        csv_filename = os.path.join('site_csvfiles', f'{site_name}_ensemble.csv')
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)  # Create directory if not exists
        site_df.to_csv(csv_filename, index=False)

    return var_values_by_site_time