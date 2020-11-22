# DSO670-RSoOPAbUP-reproduction
This repository contains the code for roproducing the experiments in the paper [Ben-Tal et al. (2013)](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.1120.1641?casa_token=tIXCfPTRdUgAAAAA:7WP0rZS-5KAzoThHcntaHkRB_UaFQxUO5p87ddg1iDBjWO6fcWU-sXuKpci2vRTvsFmV3Q02n9ss7w) and several relevant experiments whose details can be found in the [report document](https://github.com/yeyingxiao/DSO670-RSoOPAbUP-reproduction/blob/main/DSO_670_Project.pdf).

This code requires the following:
- Python 3 (with numpy and pandas)
- Gurobi and python module gurobipy (free for academic use)

## General Outline
- **model.py:** contains functions for construction of the robust model and nonrobust model.
- **data.py:** contains functions for importing data, simulation and calculating important parameters.
- **experiments.py:** contains functions for evaluation, plotting and experiments implementing.
- **main.py:** contains the main function for setting the default parameter and running the experiments.
