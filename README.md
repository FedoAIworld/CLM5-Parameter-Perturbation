# CLM5 Parameter Perturbation
This repository provides scripts for generating ensemble realizations with perturbed parameters in the [Community Land Model version 5.0 (CLM5)](https://github.com/FedoAIworld/clm5_0/tree/release-clm5.0-add-params). 

## Perturbation Experiments
1. **Perturbing Atmospheric Forcings**
   - **Script:** `forcing_perturbation.py`
   - **Description:** Generates ensemble members by applying variations to atmospheric forcing variables such as temperature, precipitation, solar radiation, wind speed, and relative humidity.

2. **Perturbing Soil Parameters**
   - **Script:** `soil_params_perturb.py`
   - **Description:** Perturbs soil textures and hydraulic parameters based on predictive statistics from Cosby et al. (1984), enabling realistic variability in soil-related dynamics.

3. **Perturbing Vegetation Parameter**
   - **Scripts:**
       - `veg_params_perturb.py`
       - `veg_params_perturb_be_bra.py` (for mixed ecosystems)
   - **Description:** Applies perturbations to vegetation parameters for each PFT, ensuring diverse ecosystem representations.
  
## Citation
Cite code: [![DOI](https://zenodo.org/badge/811938241.svg)](https://doi.org/10.5281/zenodo.14199213)

## Contact
For questions, feedback, or contributions, feel free to contact me:
- [Fernand B. Eloundou](https://github.com/FedoAIworld)
