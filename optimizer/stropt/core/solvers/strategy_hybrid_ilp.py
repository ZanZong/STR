from enum import Enum
import logging
import math
import os
from re import M
from stropt.core.utils.swapping import SwapControler, prun_q_opt
from typing import Dict, Any, Optional

import numpy as np
# noinspection PyPackageRequirements
from gurobipy import GRB, Model, quicksum, and_

import stropt.core
from stropt.core.dfgraph import DFGraph
from stropt.core.schedule import ScheduledResult, ILPAuxData, SchedulerAuxData
from stropt.core.utils.definitions import PathLike
from stropt.core.utils.solver_common import SOLVER_DTYPE, SolverTarget
from stropt.core.utils.scheduler import schedule_from_rspq
from stropt.core.utils.result_check import *
from stropt.core.enum_strategy import SolveStrategy, ImposedSchedule
from stropt.core.utils.timer import Timer



# A hybrid solver that combines swapping and re-computing, drafted from `strategy_optimal_ilp.py`
class HybridILPSolver:
    def __init__(self, g: DFGraph, budget: int, eps_noise=None, seed_s=None, integral=True,
                 imposed_schedule: ImposedSchedule=ImposedSchedule.FULL_SCHEDULE, solve_r=True,
                 write_model_file: Optional[PathLike] = None, gurobi_params: Dict[str, Any] = None, 
                 target=SolverTarget.MIN_COMPUTE, cpu_fwd_factor=2, batch_size_min=None):
        self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT = 300  # todo (paras): read this from gurobi_params
        self.gurobi_params = gurobi_params
        self.num_threads = self.gurobi_params.get('Threads', 1)
        self.model_file = write_model_file
        self.seed_s = seed_s
        self.integral = integral
        self.imposed_schedule = imposed_schedule
        self.solve_r = solve_r
        self.eps_noise = eps_noise
        self.budget = budget
        self.g: DFGraph = g
        self.solve_time = None
        self.target = target
        self.cpu_fwd_factor = cpu_fwd_factor
        self.batch_size_min = batch_size_min

        if not self.integral:
            assert not self.solve_r, "Can't solve for R if producing a fractional solution"
        assert not (target == SolverTarget.MAX_BATCHSIZE and not integral), \
            "Approximation is not allowed when your target is maximum batch size."

        self.init_constraints = []  # used for seeding the model

        self.m = Model("checkpointmip_gc_{}_{}".format(self.g.size, self.budget))
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        self.swap_control = SwapControler(10E9, g.cost_cpu, g.cost_ram)

        T = self.g.size
        print(f"graph size={T}")
        self.ram_gcd = self.g.ram_gcd(self.budget)
        if self.integral:
            self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
            self.S = self.m.addVars(T, T, name="S", vtype=GRB.BINARY)
            self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.BINARY)
            self.P = self.m.addVars(T, T, name="P", vtype=GRB.BINARY)
            self.Q = self.m.addVars(T, T, name="Q", vtype=GRB.BINARY)
        else:
            # approximation
            self.R = self.m.addVars(T, T, name="R", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0) # lower/upper bound
            self.S = self.m.addVars(T, T, name="S", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            self.P = self.m.addVars(T, T, name="P", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            self.Q = self.m.addVars(T, T, name="Q", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
        self.U = self.m.addVars(T, T, name="U", lb=0.0, ub=float(budget) / self.ram_gcd) # divided by gcd to scale down the upper bound

        for x in range(T):
            for y in range(T):
                # set memory constraints
                self.m.addLConstr(self.U[x, y], GRB.GREATER_EQUAL, 0)
                self.m.addLConstr(self.U[x, y], GRB.LESS_EQUAL, float(budget) / self.ram_gcd)
        # Only using for finding maximum batch with the upper bound of computation overhead
        if self.target == SolverTarget.MAX_BATCHSIZE:
            self.BS = self.m.addVar(lb=batch_size_min, ub=1024 * 8, name="BatchSize")
        else:
            self.BS = 1 # batch size has been calculated into permute_ram for normal opt

    def build_model(self):
        T = self.g.size
        dict_val_div = lambda cost_dict, divisor: {k: v / divisor for k, v in cost_dict.items()}
        permute_ram = dict_val_div(self.g.cost_ram, self.ram_gcd) # scale down memory cost of each node
        budget = self.budget / self.ram_gcd # scaled extra memory that required

        permute_eps = lambda cost_dict, eps: {k: v * (1. + eps * np.random.randn()) for k, v in cost_dict.items()}
        permute_cpu = dict_val_div(self.g.cost_cpu, self.g.cpu_gcd()) # scale down runtime of each node
        if self.eps_noise:
            permute_cpu = permute_eps(permute_cpu, self.eps_noise) # add noise for runtime's measurement

        with Timer("Gurobi model construction", extra_data={'T': str(T), 'budget': str(budget)}):
            with Timer("Objective construction", extra_data={'T': str(T), 'budget': str(budget)}):
                # seed solver with a baseline strategy
                if self.seed_s is not None:
                    for x in range(T):
                        for y in range(T):
                            if self.seed_s[x, y] < 1:
                                self.init_constraints.append(self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                    self.m.update()

                # define objective function
                if self.target == SolverTarget.MAX_BATCHSIZE:
                    self.m.setObjective(self.BS, GRB.MAXIMIZE)
                else:
                    self.m.setObjective(quicksum(
                        self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)),
                        GRB.MINIMIZE)

            with Timer("Variable initialization", extra_data={'T': str(T), 'budget': str(budget)}):
                if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
                    self.m.addLConstr(quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.S[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)
                    # upper-triangular is zero
                    self.m.addLConstr(quicksum(self.P[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.Q[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                    for t in range(T):
                        for i in range(t):
                            ss = self.swap_control.swap_start_stage(t, i)
                            sf = self.swap_control.swap_finish_stage(t, i)
                            if ss is None or sf is None:
                                self.m.addLConstr(self.P[t, i], GRB.EQUAL, 0)
                                self.m.addLConstr(self.Q[t, i], GRB.EQUAL, 0)

                elif self.imposed_schedule == ImposedSchedule.COVER_ALL_NODES:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    for i in range(T):
                        self.m.addLConstr(quicksum(self.R[t, i] for t in range(T)), GRB.GREATER_EQUAL, 1)
                elif self.imposed_schedule == ImposedSchedule.COVER_LAST_NODE:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    # note: the integrality gap is very large as this constraint
                    # is only applied to the last node (last column of self.R).
                    self.m.addLConstr(quicksum(self.R[t, T-1] for t in range(T)), GRB.GREATER_EQUAL, 1)
        
            with Timer("Correctness constraints", extra_data={'T': str(T), 'budget': str(budget)}):
                # ensure all computations are possible
                for (u, v) in self.g.edge_list:
                    for t in range(T):
                        # start_stage = self.swap_control.swap_start_stage(t, u)
                        # if start_stage is not None:
                            # self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.Q[start_stage, u] + self.R[t, u] + self.S[t, u])
                        # else:
                            self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.R[t, u] + self.S[t, u])
                # ensure all checkpoints are in memory
                for t in range(T - 1):
                    for i in range(T):
                        start_stage = self.swap_control.swap_start_stage(t, i)
                        if start_stage is not None:
                            self.m.addLConstr(self.S[t + 1, i], GRB.LESS_EQUAL, self.Q[start_stage, i] + self.S[t, i] + self.R[t, i])
                        else:
                            self.m.addLConstr(self.S[t + 1, i], GRB.LESS_EQUAL, self.S[t, i] + self.R[t, i])

            with Timer("Constraint: swapping correctness", extra_data={'T': str(T), 'budget': str(budget)}):
                # 1. only swap-out once
                for i in range(T):
                    self.m.addLConstr(quicksum(self.P[t, i] for t in range(T)), GRB.LESS_EQUAL, 1)
                # 2. need to swap out first
                for t in range(T):
                    for i in range(t):
                        for k in range(t + 1):
                            self.m.addLConstr(self.P[t, i] + self.Q[k, i] - 1, GRB.LESS_EQUAL, 0)
                # 3. allow swap-in up to eta times for each tensor, zero P => zero Q
                n_eta = 5
                for i in range(T):
                    self.m.addLConstr(quicksum(self.P[t, i] for t in range(T)), GRB.LESS_EQUAL, quicksum(self.Q[t, i] for t in range(T)))
                    self.m.addLConstr(n_eta * quicksum(self.P[t, i] for t in range(T)), GRB.GREATER_EQUAL, quicksum(self.Q[t, i] for t in range(T)))
                #    constraint the total times of using swap-in
                # self.m.addLConstr(quicksum(self.Q[t, i] for t in range(T) for i in range(T)), GRB.LESS_EQUAL, 2 * n_eta)
                # 4. P and Q bandwidth usage exclusive separately
                # 4.1 column constraint
                for t in range(T):
                    for i in range(t):
                        finish_point = self.swap_control.swap_finish_stage(t, i)
                        if finish_point is not None:
                            self.m.addLConstr(self.Q[t, i], GRB.LESS_EQUAL, quicksum(self.Q[m, i] for m in range(t, finish_point + 1)))
                            self.m.addLConstr(1 + (1 - self.Q[t, i]) * (finish_point-t-1),
                                                            GRB.GREATER_EQUAL, quicksum(self.Q[m, i] for m in range(t, finish_point + 1)))
                            self.m.addLConstr(self.P[t, i], GRB.LESS_EQUAL, quicksum(self.P[m, i] for m in range(t, finish_point + 1)))
                            self.m.addLConstr(1 + (1 - self.P[t, i]) * (finish_point-t-1),
                                                            GRB.GREATER_EQUAL, quicksum(self.P[m, i] for m in range(t, finish_point + 1)))
                # 4.2 row constraint
                for t in range(T):
                    for i in range(t):
                        finish_point = self.swap_control.swap_finish_stage(t, i)
                        if finish_point is not None:
                            self.m.addLConstr(self.Q[t, i], GRB.LESS_EQUAL, quicksum(self.Q[m, k] for m in range(t, finish_point + 1) for k in range(T)))
                            self.m.addLConstr(1 + (1 - self.Q[t, i]) * (finish_point-t+1),
                                                            GRB.GREATER_EQUAL, quicksum(self.Q[m, k] for m in range(t, finish_point + 1) for k in range(T)))
                            self.m.addLConstr(self.P[t, i], GRB.LESS_EQUAL, quicksum(self.P[m, k] for m in range(t, finish_point + 1) for k in range(T)))
                            self.m.addLConstr(1 + (1 - self.P[t, i]) * (finish_point-t+1),
                                                            GRB.GREATER_EQUAL, quicksum(self.P[m, k] for m in range(t, finish_point + 1) for k in range(T)))                                 
                
                # 4.3 tensor existence during swap out and swapping surrogates recomputing
                for t in range(T):
                    for i in range(t):
                        start_stage = self.swap_control.swap_start_stage(t, i)
                        if start_stage is not None:
                            # self.m.addLConstr(self.S[t, i] + self.Q[start_stage, i], GRB.LESS_EQUAL, 1)
                            self.m.addLConstr(self.S[t, i], GRB.GREATER_EQUAL, self.P[start_stage, i])
                            # self.m.addLConstr(self.R[t, i] + self.S[t, i] + self.Q[start_stage, i], GRB.LESS_EQUAL, 1)


            with Timer("Constraint: initialize memory usage (includes spurious checkpoints)",
                       extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    # Swap-out tensors in P will be calculated in S.
                    # Memory occupied by tensors in Q means unfinished
                    # swap-in tensors, which is not calculated in S.
                    candidates = self.swap_control.swap_candidate(t)
                    if self.target == SolverTarget.MAX_BATCHSIZE:
                        self.m.addConstr(self.U[t, 0], GRB.EQUAL,
                                        self.BS * (self.R[t, 0] * permute_ram[0] + quicksum(
                                            self.S[t, i] * permute_ram[i] for i in range(T)) \
                                                + quicksum(self.Q[t_prime, i_prime] * permute_ram[i_prime] for (t_prime, i_prime) in candidates)))
                    else:
                        self.m.addLConstr(self.U[t, 0], GRB.EQUAL,
                                        self.BS * (self.R[t, 0] * permute_ram[0] + quicksum(
                                            self.S[t, i] * permute_ram[i] for i in range(T)) \
                                                + quicksum(self.Q[t_prime, i_prime] * permute_ram[i_prime] for (t_prime, i_prime) in candidates)))
            with Timer("Constraint: memory recurrence", extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for k in range(T - 1):
                        mem_freed = self.BS * quicksum(
                            permute_ram[i] * self.Free_E[t, eidx] for (eidx, i) in self.g.predecessors_indexed(k))
                        allocated = self.BS * self.R[t, k + 1] * permute_ram[k + 1]
                        if self.target == SolverTarget.MAX_BATCHSIZE:
                            self.m.addConstr(self.U[t, k + 1], GRB.EQUAL,
                                            self.U[t, k] - mem_freed + allocated)
                        else:
                            self.m.addLConstr(self.U[t, k + 1], GRB.EQUAL,
                                            self.U[t, k] - mem_freed + allocated)
    
             # define memory constraints
            def _num_hazards(t, i, k):
                if t + 1 < T:
                    return 1 - self.R[t, k] + self.S[t + 1, i] + quicksum(
                        self.R[t, j] for j in self.g.successors(i) if j > k)
                return 1 - self.R[t, k] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)

            def _max_num_hazards(t, i, k):
                num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
                if t + 1 < T:
                    return 2 + num_uses_after_k
                return 1 + num_uses_after_k 

            with Timer("Constraint: upper bound for 1 - Free_E",
                       extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(1 - self.Free_E[t, eidx], GRB.LESS_EQUAL, _num_hazards(t, i, k))
            with Timer("Constraint: lower bound for 1 - Free_E",
                       extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(_max_num_hazards(t, i, k) * (1 - self.Free_E[t, eidx]),
                                          GRB.GREATER_EQUAL, _num_hazards(t, i, k))
            
            # Add upper bound of maximum compute overhead
            if self.target == SolverTarget.MAX_BATCHSIZE:
                with Timer("Constraint: Upper bound for maximum compute costs"):
                    compute_fwd = sum([permute_cpu[i] for i in self.g.vfwd])
                    bwd_compute = sum([permute_cpu[i] for i in self.g.v if i not in self.g.vfwd])
                    max_compute = (self.cpu_fwd_factor * compute_fwd + bwd_compute)
                    print(f"Solver using compute overhead ceiling of {max_compute}")
                    self.m.addLConstr(quicksum(self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)) <= max_compute, name="limit_cpu")

                #==additional, unnecessary constraints to increase tightness of relaxation
                with Timer("Unnecessary Constraint: sum_k Free_{t,i,k} <= 1"):
                    for t in range(T):
                        for i in range(T):
                            self.m.addLConstr(quicksum(self.Free_E[t, eidx] for eidx, (j, _) in enumerate(self.g.edge_list) if i == j), GRB.LESS_EQUAL, 1)
                with Timer("Unnecessary Constraint: Free_{t,i,k} <= 1 - S_{t+1, i}"):
                    for t in range(T-1):
                        for eidx, (i, k) in enumerate(self.g.edge_list):
                            self.m.addLConstr(self.Free_E[t, eidx] + self.S[t+1, i], GRB.LESS_EQUAL, 1)
                with Timer("Unnecessary Constraint: Free_{t,i,k} <= 1 - R_{t, j}"):
                    for t in range(T):
                        for eidx, (i, k) in enumerate(self.g.edge_list):
                            for j in self.g.successors(i):
                                if j > k:
                                    self.m.addLConstr(self.Free_E[t, eidx] + self.R[t, j], GRB.LESS_EQUAL, 1)
                #==end additional constraints

        if self.model_file is not None and self.g.size < 200:  # skip for big models to save runtime
            with Timer("Saving model", extra_data={'T': str(T), 'budget': str(budget)}):
                self.m.write(self.model_file)
        return None  # return value ensures ray remote call can be chained

    def solve(self):
        T = self.g.size
        with Timer('Gurobi model optimization', extra_data={'T': str(T), 'budget': str(self.budget)}):
            if self.seed_s is not None:
                self.m.Params.TimeLimit = self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT
                self.m.optimize()
                if self.m.status == GRB.INFEASIBLE:
                    print(f"Infeasible ILP seed at budget {self.budget:.2E}")
                self.m.remove(self.init_constraints)
            self.m.Params.TimeLimit = self.gurobi_params.get('TimeLimit', 0)
            self.m.message("\n\nRestarting solve\n\n")
            with Timer("ILPSolve") as solve_ilp:
                self.m.optimize()
            self.solve_time = solve_ilp.elapsed
            print(f"Solve time of hybrid optimization with approximation={not self.integral}: {self.solve_time}s")
            self.m.write(self.gurobi_params['LogFile'][:str(self.gurobi_params['LogFile']).rindex('/')] + "/model.lp")

        infeasible = (self.m.status == GRB.INFEASIBLE)
        if infeasible:
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")

        if self.m.solCount < 1:
            raise ValueError(f"Model status is {self.m.status} (not infeasible), but solCount is {self.m.solCount}")

        print("Convert variables to result array...")
        Rout = np.zeros((T, T), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Sout = np.zeros((T, T), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Uout = np.zeros((T, T), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Free_Eout = np.zeros((T, len(self.g.edge_list)), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE)
        Pout = np.zeros((T, T), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Qout = np.zeros((T, T), dtype=stropt.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        solver_dtype_cast = int if self.integral else float
        
        try:
            for t in range(T):
                for i in range(T):
                        Rout[t][i] = solver_dtype_cast(self.R[t, i].X)
                        Sout[t][i] = solver_dtype_cast(self.S[t, i].X)
                        Uout[t][i] = self.U[t, i].X * self.ram_gcd
                        Pout[t][i] = solver_dtype_cast(self.P[t, i].X)
                        Qout[t][i] = solver_dtype_cast(self.Q[t, i].X)
                        if self.target == SolverTarget.MAX_BATCHSIZE:
                            BSout = self.BS.X
                for e in range(len(self.g.edge_list)):
                        Free_Eout[t][e] = solver_dtype_cast(self.Free_E[t, e].X)
        except AttributeError as e:
            logging.exception(e)
            return None, None, None, None, None

        if self.target == SolverTarget.MAX_BATCHSIZE:
            return BSout
        else:
            return Rout, Sout, Uout, Free_Eout, Pout, Qout
    
    def format_matrix(self, matrix, label: str, approx_fmt="%.3f"):
        log_path = self.gurobi_params['LogFile'][:str(self.gurobi_params['LogFile']).rindex('/')]
        log_file_name = log_path + f"/{label}"
        if self.integral:
            out_fmt = "%.6e" if label == "U" else "%i"
        else:
            out_fmt = "%.6e" if label == "U" else approx_fmt
        np.savetxt(log_file_name, matrix, fmt=out_fmt)

    def dump_swap_finish_stage(self):
        T = self.g.size
        swap_finish_mat = np.zeros((T, T), dtype=np.integer)
        swap_start_mat = np.zeros((T, T), dtype=np.integer)
        for t in range(T):
            for i in range(T):
                sf = self.swap_control.swap_finish_stage(t, i)
                ss = self.swap_control.swap_start_stage(t, i)
                swap_finish_mat[t, i] = sf if sf is not None else -1
                swap_start_mat[t, i] = ss if ss is not None else -1
        return swap_finish_mat, swap_start_mat

def solve_hybrid_ilp(g: DFGraph, budget: int, seed_s: Optional[np.ndarray] = None, approx=True,
                     imposed_schedule: ImposedSchedule=ImposedSchedule.FULL_SCHEDULE, solve_r=False,
                     time_limit: Optional[int] = None, write_log_file: Optional[PathLike] = None, print_to_console=True,
                     write_model_file: Optional[PathLike] = None, eps_noise=0.01, solver_cores=os.cpu_count()):
    """
    Memory-accurate solver with garbage collection.
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
                          solve_r=solve_r, write_model_file=write_model_file)
    ilpsolver.build_model()
    try:
        r, s, u, free_e, p, q = ilpsolver.solve()
        
        pruned_Qout = prun_q_opt(ilpsolver.swap_control, q, s)
        
        if not check_compute_correctness(g, s, r):
            # TODO check if there are some bugs usually doesn't show up
            print("Need corrected again!!!!")
            from stropt.core.utils.approximate_hybrid import fine_grained_approx
            r, s, p_appro, pruned_Qout = fine_grained_approx(g=g, sc=ilpsolver.swap_control, \
                    r=r, s=s, p=p, q=q, u=u, mem_budget=ilpsolver.budget*ilpsolver.ram_gcd)

        ilpsolver.format_matrix(r, "R")
        ilpsolver.format_matrix(s, "S")
        ilpsolver.format_matrix(u, "U")
        ilpsolver.format_matrix(free_e, "Free_Eout")
        ilpsolver.format_matrix(p, "P")
        ilpsolver.format_matrix(q, "Q")
        ilpsolver.format_matrix(pruned_Qout, "PrunedQ")
        swap_finish_mat, swap_start_mat = ilpsolver.dump_swap_finish_stage()
        ilpsolver.format_matrix(swap_finish_mat, "SFMat")
        ilpsolver.format_matrix(swap_start_mat, "SSMat")
        ilp_feasible = True
    except ValueError as e:
        logging.exception(e)
        r, s, q, u, free_e, pruned_Qout = (None, None, None, None, None, None)
        ilp_feasible = False
    ilp_aux_data = ILPAuxData(U=u, Free_E=free_e, ilp_approx=approx, ilp_time_limit=time_limit, ilp_eps_noise=eps_noise,
                              ilp_num_constraints=ilpsolver.m.numConstrs, ilp_num_variables=ilpsolver.m.numVars,
                              ilp_imposed_schedule=imposed_schedule)
    schedule, aux_data = schedule_from_rspq(g, r, s, pruned_Qout)
    return ScheduledResult(
        solve_strategy=SolveStrategy.MIXED_ILP_OPTIMAL,
        solver_budget=budget,
        feasible=ilp_feasible,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=ilpsolver.solve_time,
        ilp_aux_data=ilp_aux_data,
    )


if __name__ == '__main__':
    pass