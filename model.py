from gurobipy import *
import pandas as pd
import numpy as np


def robust_solution_optimizer(
    c_vec, v_vec, s_vec, l_vec, Q_mat,
    d_vec, budget, rho, f_div_type="Hellinger",
    problem_type="average", time_limit=600, log=1
):
    # validate input
    assert f_div_type in ["Chi2", "ModifiedChi2", "Hellinger"]
    assert problem_type in ["average", "worst_case"]

    I = d_vec.shape[0]  # number of scenarios
    J = Q_mat.shape[0]  # number of items
    # d_vec = list(d_vec)

    # create the model and set the parameters
    m = Model("Robust")
    # m.params.TimeLimit = time_limit
    m.params.logtoconsole = log

    # the objective
    tau = m.addVars(J, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="tau")
    if problem_type == "average":
        m.setObjective(quicksum(tau[j] for j in range(J)), sense=GRB.MAXIMIZE)
    elif problem_type == "worst_case":
        min_tau = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="worst_return")
        m.addGenConstrMin(min_tau, tau)
        m.setObjective(min_tau, sense=GRB.MAXIMIZE)

    # decision variables
    # order quantity
    q_vars = m.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="order_quantity")
    # lagrangian variables
    lambda_vars = m.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="lambda")
    eta_vars = m.addVars(J, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="eta")
    # artificial variables
    y_vars = m.addVars(J, I, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")
    # y_vars = m.addVars(J, I, vtype=GRB.CONTINUOUS, name="y")
    if f_div_type == "ModifiedChi2":
        z_vars = m.addVars(J, I, vtype=GRB.CONTINUOUS, lb=0, name="z")
    # f_{ij}(Q_{j})
    f_fun = m.addVars(J, I, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="f")

    # add constraints
    # budget constraint
    m.addConstr(
        quicksum(c_vec[j] * q_vars[j] for j in range(J)) <= budget,
        name="budget_constr"
    )
    for j in range(J):
        # epigraph constraint
        if f_div_type == "ModifiedChi2":
            m.addConstr(
                tau[j] + c_vec[j] * q_vars[j] + eta_vars[j] + lambda_vars[j] * (rho[j] - 1)
                + 1/4 * quicksum(Q_mat[j, i] * y_vars[j, i] for i in range(I))
                <= 0,
                name=f"robust_constr_{j}"
            )
        elif f_div_type == "Hellinger":
            m.addConstr(
                tau[j] + c_vec[j] * q_vars[j] + eta_vars[j] + lambda_vars[j] * (rho[j] - 1)
                + quicksum(Q_mat[j, i] * y_vars[j, i] for i in range(I))
                <= 0,
                name=f"robust_constr_{j}"
            )
        elif f_div_type == "Chi2":
            m.addConstr(
                tau[j] + c_vec[j] * q_vars[j] + eta_vars[j] + lambda_vars[j] * (rho[j] + 2)
                - 2 * quicksum(Q_mat[j, i] * y_vars[j, i] for i in range(I))
                <= 0,
                name=f"robust_constr_{j}"
            )

        for i in range(I):
            # function f when order > demand
            m.addConstr(
                f_fun[j, i]
                <= s_vec[j] * q_vars[j] - d_vec[i] * (s_vec[j] - v_vec[j]),
                name=f"f_Q>d_{j}_{i}"
            )
            # function f when order < demand
            m.addConstr(
                f_fun[j, i]
                <= -l_vec[j] * d_vec[i] + q_vars[j] * (l_vec[j] + v_vec[j]),
                name=f"f_d>Q_{j}_{i}"
            )
            # con1 = m.addVar(vtype=GRB.CONTINUOUS, name="con1" + str(i) + str(j))
            # con2 = m.addVar(vtype=GRB.CONTINUOUS, name="con2" + str(i) + str(j))
            # m.addConstr(con1 == 0.5 * lambda_vars[j] - y_vars[j, i])
            # m.addConstr(con2 == 0.5 * lambda_vars[j] + y_vars[j, i])
            # m.addConstr(
            #     z_vars[j, i] * z_vars[j, i] + con1 * con1
            #     <= con2 * con2
            # )
            if f_div_type == "ModifiedChi2":
                con1 = m.addVar(vtype=GRB.CONTINUOUS)
                con2 = m.addVar(vtype=GRB.CONTINUOUS)
                # m.addConstr(con1 == 0.5 * (lambda_vars[j] -  y_vars[j, i]))
                # m.addConstr(con2 == 0.5 * (lambda_vars[j] +  y_vars[j, i]))
                # m.addQConstr(
                #     z_vars[j, i] * z_vars[j, i] + con1 * con1 <= con2 * con2,
                #     name=f"soc_{j}_{i}"
                # )
                m.addConstr(con1 == lambda_vars[j])
                m.addConstr(con2 == y_vars[j, i])
                m.addConstr(
                    z_vars[j, i] * z_vars[j, i] <= con1 * con2,
                    name=f"soc_{j}_{i}"
                )
                m.addConstr(
                    z_vars[j, i] + f_fun[j, i] + eta_vars[j] - 2 * lambda_vars[j] >= 0
                )
            elif f_div_type == "Hellinger":
                con = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
                con1 = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
                con2 = m.addVar(vtype=GRB.CONTINUOUS)
                m.addConstr(con == f_fun[j, i] + eta_vars[j])
                m.addConstr(con1 == 0.5 * (y_vars[j, i] - lambda_vars[j] - con))
                m.addConstr(con2 == 0.5 * (y_vars[j, i] + lambda_vars[j] + con))
                m.addConstr(
                    lambda_vars[j] * lambda_vars[j] + con1 * con1 <= con2 * con2,
                    name=f"soc_{j}_{i}"
                )
                # m.addConstr(con == f_fun[j, i] + eta_vars[j])
                # m.addConstr(con1 == lambda_vars[j] + con)
                # # m.addConstr(con2 == 0.5 * (y_vars[j, i] + lambda_vars[j] + con))
                # m.addConstr(
                #     lambda_vars[j] * lambda_vars[j] <= y_vars[j, i] * con1,
                #     name=f"soc_{j}_{i}"
                # )
                m.addConstr(
                    f_fun[j, i] + eta_vars[j] + lambda_vars[j] >= 0
                )
            elif f_div_type == "Chi2":
                con = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
                con1 = m.addVar(vtype=GRB.CONTINUOUS)
                con2 = m.addVar(vtype=GRB.CONTINUOUS)
                # m.addConstr(con == f_fun[j, i] + eta_vars[j])
                # m.addConstr(con1 == 0.5 * con)
                # m.addConstr(con2 == 0.5 * (2 * lambda_vars[j] - con))
                # m.addConstr(
                #     y_vars[j, i] * y_vars[j, i] + con1 * con1 <= con2 * con2,
                #     name=f"soc_{j}_{i}"
                # )
                m.addConstr(con == f_fun[j, i] + eta_vars[j])
                m.addConstr(con1 == -0.5 * con)
                m.addConstr(con2 == 0.5 * (2 * lambda_vars[j] + con))
                m.addConstr(
                    y_vars[j, i] * y_vars[j, i] + con1 * con1 <= con2 * con2,
                    name=f"soc_{j}_{i}"
                )
                m.addConstr(
                    f_fun[j, i] + eta_vars[j] + lambda_vars[j] >= 0
                )

    m.update()
    m.optimize()

    return m


def nonrobust_solution_optimizer(
    c_vec, v_vec, s_vec, l_vec, Q_mat, d_vec, budget,
    problem_type="average", time_limit=600, log=1
):
    # validate input
    assert problem_type in ["average", "worst_case"]

    I = d_vec.shape[0]  # number of scenarios
    J = Q_mat.shape[0]  # number of items
    # d_vec = list(d_vec)

    # create the model and set the parameters
    m = Model("Nonrobust")
    # m.params.TimeLimit = time_limit
    m.params.logtoconsole = log

    # the objective
    tau = m.addVars(J, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="tau")
    if problem_type == "average":
        m.setObjective(quicksum(tau[j] for j in range(J)), sense=GRB.MAXIMIZE)
    elif problem_type == "worst_case":
        min_tau = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="worst_return")
        m.addGenConstrMin(min_tau, tau)
        m.setObjective(min_tau, sense=GRB.MAXIMIZE)

    # decision variables
    # order quantity
    q_vars = m.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="order_quantity")
    f_fun = m.addVars(J, I, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="f")

    # add constraints
    # budget constraint
    m.addConstr(
        quicksum(c_vec[j] * q_vars[j] for j in range(J)) <= budget,
        name="budget_constr"
    )
    for j in range(J):
        # epigraph constraint
        m.addConstr(
            tau[j] + c_vec[j] * q_vars[j]
            - quicksum(Q_mat[j, i] * f_fun[j, i] for i in range(I))
            <= 0,
            name=f"epigraph_constr_{j}"
        )

        for i in range(I):
            # function f when order > demand
            m.addConstr(
                f_fun[j, i]
                <= s_vec[j] * q_vars[j] - d_vec[i] * (s_vec[j] - v_vec[j]),
                name=f"f_Q>d_{j}_{i}"
            )
            # function f when order < demand
            m.addConstr(
                f_fun[j, i]
                <= -l_vec[j] * d_vec[i] + q_vars[j] * (l_vec[j] + v_vec[j]),
                name=f"f_d>Q_{j}_{i}"
            )

    m.update()
    m.optimize()

    return m
