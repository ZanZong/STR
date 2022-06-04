import logging
import math
import os
import time
from typing import Optional
import numpy as np
from stropt.core.dfgraph import DFGraph
from stropt.core.enum_strategy import SolveStrategy, ImposedSchedule
from stropt.core.schedule import ILPAuxData, ScheduledResult
from stropt.core.solvers.strategy_hybrid_ilp import HybridILPSolver
from stropt.core.utils.swapping import SwapControler, prun_q_opt
from stropt.core.utils.definitions import PathLike
from stropt.core.utils.scheduler import schedule_from_rspq
import stropt.core.utils.solver_common as solver_common
from stropt.core.utils.approximate_hybrid import fine_grained_approx, fill_p
# from stropt.core.utils.approximate_graph_par import graph_partition
from stropt.core.solvers.graph_reducer import simplify, recover


def solve_hybrid_approximate_lp(g: DFGraph, budget: int, seed_s: Optional[np.ndarray] = None, approx=True,
                     imposed_schedule: ImposedSchedule=ImposedSchedule.FULL_SCHEDULE, solve_r=False,
                     time_limit: Optional[int] = None, write_log_file: Optional[PathLike] = None, print_to_console=True,
                     write_model_file: Optional[PathLike] = None, eps_noise=0.01, solver_cores=os.cpu_count()):
    """
    Approximation solver with constraints relation, i.e., using the RecursiveSourceTracing algorithm.

    :param g: DFGraph -- graph definition extracted from model
    :param budget: int -- budget constraint for solving
    :param seed_s: np.ndarray -- optional parameter to set warm-start for solver, defaults to empty S
    :param approx: bool -- set true to return as soon as a solution is found that is within 1% of optimal
    :param imposed_schedule -- selects a set of constraints on R and S that impose a schedule or require some nodes to be computed
    :param solve_r -- if set, solve for the optimal R 
    :param time_limit: int -- time limit for solving in seconds
    :param write_log_file: if set, log gurobi to this file
    :param print_to_console: if set, print gurobi logs to the console
    :param write_model_file: if set, write output model file to this location
    :param eps_noise: float -- if set, inject epsilon noise into objective weights, default 0.5%
    :param solver_cores: int -- if set, use this number of cores for ILP solving
    """
    param_dict = {'LogToConsole': 1 if print_to_console else 0,
                  'LogFile': str(write_log_file) if write_log_file is not None else "",
                  'Threads': solver_cores,
                  'TimeLimit': math.inf if time_limit is None else time_limit,
                  'OptimalityTol': 1e-2 if approx else 1e-4,
                  'IntFeasTol': 1e-3 if approx else 1e-5,
                  'Presolve': 2,
                  'StartNodeLimit': 10000000}
    ilpsolver = HybridILPSolver(g, budget, gurobi_params=param_dict, seed_s=seed_s,
                          eps_noise=eps_noise, imposed_schedule=imposed_schedule,
                          solve_r=solve_r, write_model_file=write_model_file, integral=False)
    ilpsolver.build_model()
    try:
        r, s, u, free_e, p, q = ilpsolver.solve()
        ilpsolver.format_matrix(r, "R")
        ilpsolver.format_matrix(s, "S")
        ilpsolver.format_matrix(u, "U")
        ilpsolver.format_matrix(free_e, "Free_Eout")
        ilpsolver.format_matrix(p, "P")
        ilpsolver.format_matrix(q, "Q")
        # Approximation
        T = g.size
        r_appro, s_appro, p_appro, q_appro = fine_grained_approx(g=g, sc=ilpsolver.swap_control, \
                    r=r, s=s, p=p, q=q, u=u, mem_budget=ilpsolver.budget*ilpsolver.ram_gcd)
        # pruned_q = ilpsolver.prun_q_opt(q_, s_)
        # ilpsolver.format_matrix(pruned_q, "PrunedQ", approx_fmt="%i") # pruned_q is edited below
        
        ilpsolver.format_matrix(r_appro, "R-approx", approx_fmt="%i")
        ilpsolver.format_matrix(s_appro, "S-approx", approx_fmt="%i")
        ilpsolver.format_matrix(p_appro, "P-approx", approx_fmt="%i")
        ilpsolver.format_matrix(q_appro, "Q-approx", approx_fmt="%i")

        ilp_feasible = True
    except ValueError as e:
        logging.exception(e)
        r, s, q, u, free_e = (None, None, None, None, None)
        r_appro, s_appro, q_appro = (None, None, None)
        ilp_feasible = False
    ilp_aux_data = ILPAuxData(U=u, Free_E=free_e, ilp_approx=approx, ilp_time_limit=time_limit, ilp_eps_noise=eps_noise,
                              ilp_num_constraints=ilpsolver.m.numConstrs, ilp_num_variables=ilpsolver.m.numVars,
                              ilp_imposed_schedule=imposed_schedule)

    schedule, aux_data = schedule_from_rspq(g, r_appro, s_appro, q_appro)
    return ScheduledResult(
        solve_strategy=SolveStrategy.MIXED_ILP_OPTIMAL,
        solver_budget=budget,
        feasible=ilp_feasible,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=ilpsolver.solve_time,
        ilp_aux_data=ilp_aux_data,
    )



def reduced_hybrid_appro_ilp(g: DFGraph, budget: int, seed_s: Optional[np.ndarray] = None, approx='partition',
                     imposed_schedule: ImposedSchedule=ImposedSchedule.FULL_SCHEDULE, solve_r=False,
                     time_limit: Optional[int] = None, write_log_file: Optional[PathLike] = None, print_to_console=True,
                     write_model_file: Optional[PathLike] = None, eps_noise=0.01, solver_cores=os.cpu_count(), reduce_graph_size=64):
    """
    Approximation with computational graph partitioning.

    :param g: DFGraph -- graph definition extracted from model
    :param budget: int -- budget constraint for solving
    :param seed_s: np.ndarray -- optional parameter to set warm-start for solver, defaults to empty S
    :param approx: 'partition' or 'relaxtion' -- two approximation strategies
    :param imposed_schedule -- selects a set of constraints on R and S that impose a schedule or require some nodes to be computed
    :param solve_r -- if set, solve for the optimal R 
    :param time_limit: int -- time limit for solving in seconds
    :param write_log_file: if set, log gurobi to this file
    :param print_to_console: if set, print gurobi logs to the console
    :param write_model_file: if set, write output model file to this location
    :param eps_noise: float -- if set, inject epsilon noise into objective weights, default 0.5%
    :param solver_cores: int -- if set, use this number of cores for ILP solving
    """
    param_dict = {'LogToConsole': 1 if print_to_console else 0,
                  'LogFile': str(write_log_file) if write_log_file is not None else "",
                  'Threads': solver_cores,
                  'TimeLimit': math.inf if time_limit is None else time_limit,
                  'OptimalityTol': 1e-2 if approx else 1e-4,
                  'IntFeasTol': 1e-3 if approx else 1e-5,
                  'Presolve': 2,
                  'StartNodeLimit': 10000000}
    print(f"Set fused graph size {reduce_graph_size}")
    ## TODO simplify graph
    t_simplify_start = time.time()
    new_graph, fuse_handler = simplify(g, reduce_graph_size)
    t_simplify_end = time.time()
    ilpsolver = HybridILPSolver(new_graph, budget, gurobi_params=param_dict, seed_s=seed_s,
                          eps_noise=eps_noise, imposed_schedule=imposed_schedule,
                          solve_r=solve_r, write_model_file=write_model_file)
    # use relaxation
    # ilpsolver = HybridILPSolver(new_graph, budget, gurobi_params=param_dict, seed_s=seed_s,
    #                       eps_noise=eps_noise, imposed_schedule=imposed_schedule,
    #                       solve_r=solve_r, write_model_file=write_model_file, integral=False)
    ilpsolver.build_model()
    try:
        t_solve_start = time.time()
        r, s, u, free_e, p, q = ilpsolver.solve()
        t_solve_end = time.time()
        # use relaxation
        # r, s, p, q = fine_grained_approx(g=new_graph, sc=ilpsolver.swap_control, \
        #             r=r, s=s, p=p, q=q, u=u, mem_budget=ilpsolver.budget*ilpsolver.ram_gcd)
        q = prun_q_opt(ilpsolver.swap_control, q, s)

        t_recover_start = time.time()
        rec_r, rec_p, rec_q = recover(g, r, s, p, q, fuse_handler)
        rec_p = fill_p(rec_q)
        t_recover_end = time.time()
        print(f"Time cost: simplify={t_simplify_end - t_simplify_start}s, solving={t_solve_end - t_solve_start}s, recover={t_recover_end - t_recover_start}s")
        ilpsolver.format_matrix(rec_r, "R-approx", approx_fmt="%i")
        ilpsolver.format_matrix(s, "S-approx", approx_fmt="%i")
        ilpsolver.format_matrix(u, "U-approx")
        ilpsolver.format_matrix(free_e, "Free_Eout-approx", approx_fmt="%i")
        ilpsolver.format_matrix(rec_p, "P-approx", approx_fmt="%i")
        ilpsolver.format_matrix(rec_q, "Q-approx", approx_fmt="%i")
        # ilpsolver.format_matrix(pruned_Qout, "PrunedQ")
        # swap_finish_mat, swap_start_mat = ilpsolver.dump_swap_finish_stage()
        # ilpsolver.format_matrix(swap_finish_mat, "SFMat")
        # ilpsolver.format_matrix(swap_start_mat, "SSMat")
    except ValueError as e:
        logging.exception(e)
        r, s, q, u, free_e, pruned_Qout = (None, None, None, None, None, None)
    return None