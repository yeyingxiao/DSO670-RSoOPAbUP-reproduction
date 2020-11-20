#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
from data import *
from model import robust_solution_optimizer, nonrobust_solution_optimizer
from experiments import evaluate, experiment


def main():

    # import data
    c_vec, v_vec, s_vec, l_vec, Q_mat = get_data("Data.csv")
    # default settings
    budget = 1000  # budget
    d_vec = np.array([4, 8, 10])  # demand scenarios
    alpha = 0.05  # confidence level for the model
    alpha_eval = 0.05  # confidence level for the evaluation
    num_sample = 1000  # number of p-vectors for evaluation
    problem_type = "average"  # ["average", "worst_case"]
    f_div_type = "Hellinger"  # ["Chi2", "ModifiedChi2", "Hellinger"]
    print("###############################")

    # change the following codes for different experiment
    experiment(
        c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
        num_sample, alpha, alpha_eval, problem_type, f_div_type, corrected=True
    )

    # alpha_list = [0.001, 0.01, 0.05, 0.10]
    # N_list = [10, 50, 100, 200, 500, 1000]
    # for N_s in N_list:
    #     for alpha in alpha_list:
    #         experiment(
    #             c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
    #             num_sample, alpha, alpha_eval, problem_type, f_div_type, N_s, corrected=True
    #         )

    # for f_div_type in ["Chi2", "ModifiedChi2", "Hellinger"]:  # 
    for budget in [1000]:  # , 100, 250, 500, 750, 1000
        experiment(
            c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
            num_sample, alpha, alpha_eval, problem_type, f_div_type, corrected=True
        )


if __name__ == "__main__":
    main()
