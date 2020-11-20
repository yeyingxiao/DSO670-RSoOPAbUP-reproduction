import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import *
from model import robust_solution_optimizer, nonrobust_solution_optimizer


def evaluate(
    sol, p_vector, c_vec, v_vec, s_vec, l_vec, d_vec, problem_type="average"
):
    '''
    Evaluate the objective function value of a given solution.
    Return a list of objective values evaluated by each p-vector.
    '''
    obj_list = []
    for k in range(p_vector.shape[1]):
        net_profit = 0
        if problem_type == "worst_case":
            obj_min = []
        for j in range(p_vector.shape[0]):
            if problem_type == "worst_case":
                net_profit = 0
            for i in range(p_vector.shape[2]):
                net_profit += p_vector[j, k, i] * (
                    v_vec[j] * min(d_vec[i], sol[j]) +
                    s_vec[j] * max(sol[j] - d_vec[i], 0) -
                    l_vec[j] * max(d_vec[i] - sol[j], 0)
                )
            net_profit -= c_vec[j] * sol[j]
            if problem_type == "worst_case":
                obj_min.append(net_profit)
        if problem_type == "worst_case":
            obj_list.append(min(obj_min) + 2)
        else:
            obj_list.append(net_profit + 5)

    return obj_list


def plo_figure(
    range_list,
    obj_robust_mean,
    obj_robust_max,
    obj_robust_min,
    obj_nonrobust_mean,
    obj_nonrobust_max,
    obj_nonrobust_min,
    robust_obj,
    nonrobust_obj,
    problem_type,
    f_div_type,
    right_label=None
):
    '''
    Plot the figure. The structure of the figure is similar to
    the ones in Ben-Tal et al. (2013), though for different experiment,
    the x-axis may change. Also, a dataframe may be output as a csv file,
    recording the evaluation results.
    '''
    label_bbox_props = {
        'pad': 0,
        'alpha': 0.2,
        'color': 'gray',
        'linewidth': 0.2,
        'boxstyle': "round,pad=0.3",
    }
    df = pd.DataFrame(
        data={
            "N": range_list,
            "obj_robust_mean": obj_robust_mean,
            "obj_robust_max": obj_robust_max,
            "obj_robust_min": obj_robust_min,
            "obj_nonrobust_mean": obj_nonrobust_mean,
            "obj_nonrobust_max": obj_nonrobust_max,
            "obj_nonrobust_min": obj_nonrobust_min,
            "robust_obj": robust_obj,
            "nonrobust_obj": nonrobust_obj
        }
    )
    # df.to_csv(f"Figures/{problem_type}/diff_div/df_{f_div_type}.csv")
    # df.to_csv(f"Figures/{problem_type}/budget/df_{budget}.csv")
    fig, ax = plt.subplots()
    plt.plot(range_list, obj_robust_mean, linewidth='1', color='C0', label='Mean robust')
    plt.plot(range_list, obj_robust_max, color='C0', alpha=0.1, linewidth='1')
    plt.plot(range_list, obj_robust_min, color='C0', alpha=0.1, linewidth='1')
    plt.fill_between(range_list, obj_robust_min, obj_robust_max, color='C0', alpha=0.2, label='Range robust')
    plt.plot(range_list, obj_nonrobust_mean, linewidth='1', color='C1', label='Mean nonrobust')
    plt.plot(range_list, obj_nonrobust_max, color='C1', alpha=0.1, linewidth='1')
    plt.plot(range_list, obj_nonrobust_min, color='C1', alpha=0.1, linewidth='1')
    plt.fill_between(range_list, obj_nonrobust_min, obj_nonrobust_max, color='C1', alpha=0.2, label='Range nonrobust')
    plt.legend(loc='lower right', fontsize=10)
    plt.xlabel('N', fontsize=12)
    # plt.xlabel('alpha', fontsize=12)
    # plt.xlabel('N\'', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    # plt.xlim(0.0, 0.2)
    plt.xlim(0, 1000)
    # ax.set_xscale('log')
    # if problem_type == "average":
    #     plt.ylim(50, 190)
    # elif problem_type == "worst_case":
    #     plt.ylim(-15, 10)
    plt.grid(axis="y")
    if right_label is not None:
        ax.text(1.03, 0.5, f"Budget = {right_label}",
                horizontalalignment='left',
                verticalalignment='center',
                rotation=-90,
                backgroundcolor='gray',
                bbox=label_bbox_props,
                fontsize=12,
                transform=ax.transAxes,
                )
    plt.savefig(f"Figures/{problem_type}/{f_div_type}_dist.png")

    # plt.show()


def experiment(
    c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
    num_sample, alpha, alpha_eval, problem_type, f_div_type, N_s=1000, corrected=False
):

    Q_mat_copy = copy.deepcopy(Q_mat)
    m = d_vec.shape[0]  # number of scenarios
    dof = m - 1  # degree of freedom
    obj_robust_mean = []
    obj_robust_max = []
    obj_robust_min = []
    obj_nonrobust_mean = []
    obj_nonrobust_max = []
    obj_nonrobust_min = []
    robust_obj = []
    nonrobust_obj = []

    N_range = range(10, 1001, 10)
    # alpha_range = np.concatenate(
    #     (
    #         np.arange(0.0001, 0.001, 0.00003),
    #         np.arange(0.001, 0.01, 0.0003),
    #         np.arange(0.01, 0.1, 0.003),
    #         np.arange(0.1, 0.2, 0.01)
    #     )
    # )
    # alpha_range = range(10, 1001, 10)
    N = 10
    for N in N_range:

    # for N in range(100, 1001, 100):
    # for N in [10, 20, 30, 40, 50, 100, 200, 500, 1000]:
    # for N in [200, 500, 1000]:
        # Q_mat = generate_newQ(Q_mat_copy, N)
        # obj_robust_mean = []
        # obj_robust_max = []
        # obj_robust_min = []
        # obj_nonrobust_mean = []
        # obj_nonrobust_max = []
        # obj_nonrobust_min = []
        # for N_s in alpha_range:
            # print(alpha, alpha_eval)
        print(alpha, N_s)
        print("N =", N)
        print(f_div_type)
        print("budget:", budget)

        rho = compute_rho(f_div_type, N, dof, alpha, Q=Q_mat, corrected=corrected)
        # print(rho)
        m_robust = robust_solution_optimizer(
            c_vec, v_vec, s_vec, l_vec, Q_mat,
            d_vec, budget, rho, f_div_type,
            problem_type, time_limit=600, log=0
        )
        print("obj =", m_robust.objVal)
        robust_obj.append(m_robust.objVal)
        m_nonrobust = nonrobust_solution_optimizer(
            c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
            problem_type, time_limit=600, log=0
        )
        print("obj =", m_nonrobust.objVal)
        nonrobust_obj.append(m_nonrobust.objVal)

        robust_sol = []
        nonrobust_sol = []
        for j in range(c_vec.shape[0]):
            robust_sol.append(m_robust.getVarByName(f"order_quantity[{j}]").x)
            nonrobust_sol.append(m_nonrobust.getVarByName(f"order_quantity[{j}]").x)

        rho_a = compute_rho("ModifiedChi2", N, dof, alpha_eval, Q=Q_mat, corrected=False)
        p_vector = sample_p(Q_mat, num_sample, rho_a[0], m)
        # p_vector = sample_p_dist(N, Q_mat, num_sample)

        obj_robust_list = evaluate(robust_sol, p_vector, c_vec, v_vec, s_vec, l_vec, d_vec, problem_type)
        obj_nonrobust_list = evaluate(nonrobust_sol, p_vector, c_vec, v_vec, s_vec, l_vec, d_vec, problem_type)
        obj_robust_mean.append(sum(obj_robust_list) / len(obj_robust_list))
        obj_robust_max.append(max(obj_robust_list))
        obj_robust_min.append(min(obj_robust_list))
        obj_nonrobust_mean.append(sum(obj_nonrobust_list) / len(obj_nonrobust_list))
        obj_nonrobust_max.append(max(obj_nonrobust_list))
        obj_nonrobust_min.append(min(obj_nonrobust_list))
    range_list = list(N_range)
    # range_list = list(alpha_range)
    plo_figure(
        range_list,
        obj_robust_mean,
        obj_robust_max,
        obj_robust_min,
        obj_nonrobust_mean,
        obj_nonrobust_max,
        obj_nonrobust_min,
        robust_obj,
        nonrobust_obj,
        problem_type,
        f_div_type,
        right_label=None
    )
