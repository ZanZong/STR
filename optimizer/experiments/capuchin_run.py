import argparse
import numpy as np
import shutil
import pathlib
from remat.core.dfgraph import DFGraph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.core.schedule import ScheduledResult
from remat.core.utils.solver_common import gen_s_matrix_fixed_checkpoints, solve_r_opt
from remat.core.enum_strategy import SolveStrategy
from remat.core.utils.scheduler import schedule_from_rs
from remat.core.utils.timer import Timer
from experiments.common.definitions import remat_data_dir
from experiments.common.load_keras_model import get_keras_model
from experiments.common.profile.cost_model import CostModel
from remat.tensorflow2.extraction import dfgraph_from_keras
from remat.core.solvers.strategy_capuchin import Tensor, hybrid_policy


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="VGG16")
    parser.add_argument('--device-memory', nargs='+', type=int, default=[],
                        help="Target on which device size.")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

    parser.add_argument('--debug', action='store_true', help="If set, write debug files like model files.")
    parser.add_argument('--exact-ilp-solve', action='store_true', help="If set, disable approx in ILP solve.")
    parser.add_argument('--skip-ilp', action='store_true', help="If set, skip running the ILP during evaluation.")
    parser.add_argument('--ilp-time-limit', type=int, default=3600, help="Time limit for individual ILP solves, in sec")
    parser.add_argument('--hide-points', action="store_true")

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args



def cache_candidates(g:DFGraph, U: list, peak_period_threshold: float):
    """Memory usage with all checkpoints scenario, i.e., memory sufficient
    We assume that stages that larger than `peak_threshold` memory usage is the peak memory usage period.
    """
    assert len(U) == g.size, "Memory list must has the same length of graph size"
    peak_memory = max(U)
    candidates = list()
    cand_access_count = np.zeros(g.size, dtype=np.int8)
    for (u, v) in g.edge_list:
        cand_access_count[u] += 1
    for i, mem in enumerate(U):
        if cand_access_count[i] > 1 and mem >= peak_memory * peak_period_threshold:
            candidates.append(Tensor(id=i, size=g.cost_ram[i], access_count=cand_access_count[i], \
                srcs=[u for (u, v) in g.edge_list if v == i]))
    return candidates

def gen_p_q(s):
    p = np.zeros_like(s)
    q = np.zeros_like(s)
    for j in range(s.shape[1]):
        if j > (s.shape[1] / 2):
            continue
        step_into = False
        set_p = False
        set_q = False
        for i in range(s.shape[0]):
            if not step_into and s[i, j] == 1:
                step_into = True
                continue
            if i < (s.shape[0]/2) and step_into and not set_p and not set_q and s[i, j] == 0:
                p[i, j] = 1
                set_p = True
                continue
            if i > (s.shape[0]/2) and step_into and set_p and not set_q and s[i, j] == 1:
                q[i - 1, j] = 1
                set_q = True
                break
    return p, q


if __name__ == "__main__":
    args = extract_params()
    assert len(args.device_memory) > 0, "At least 1 target device memory is needed."
        
    key = "_".join(map(str, [args.model_name, args.batch_size, args.input_shape]))
    log_base = remat_data_dir() / "capuchin" / key
    shutil.rmtree(log_base, ignore_errors=True)
    pathlib.Path(log_base).mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    device_memory = [mem * 1024 * 1024 for mem in args.device_memory]
    print(f"Loading costs")

    # Build a linear regression model for each layer w.r.t batch size.
    cost_model = CostModel(model_name, "p32xlarge", log_base, quantization=5)
    cost_model.fit()
    if args.debug:
        cost_model.plot_costs()

    # load model from Keras
    print(f"Loading model {model_name}")
    model = get_keras_model(model_name, input_shape=args.input_shape)
    g = dfgraph_from_keras(model, batch_size=args.batch_size, cost_model=cost_model,
                           loss_cpu_cost=0, loss_ram_cost=(4 * args.batch_size))
    result = solve_checkpoint_all(g)
    peak_memory = result.schedule_aux_data.peak_ram
    # print(f"Schedule timeline of checkpoint all:\n {result.schedule_aux_data.mem_timeline}")
    mem_after_each_comp = [result.schedule_aux_data.mem_grid[i, i] for i in range(g.size)]
    
    results = []
    for m_device in device_memory:
        target_mem = m_device - g.cost_ram_fixed
        if peak_memory <= target_mem:
            print(f"Device memory {target_mem} is sufficent for training with batch size {args.batch_size}.")
            results.append((peak_memory, 1.0, m_device))
            continue
            
        period_threshold = 0.7 # tunnable to optimize more tensors
        candidate_set = cache_candidates(g, mem_after_each_comp, period_threshold)

        lacking_mem = peak_memory - target_mem
        print(f"Extra memory with size {lacking_mem} is required.")
        eviction_set, recomps, feasible = hybrid_policy(g, model_name, lacking_mem, candidate_set)
        if not feasible:
            print("Capuchin failed to return a solution.")
            continue
        print(f"Get a solution with {len(eviction_set)} swapping tensors and {len(recomps)} recomputing tensors.")

        last_access = np.zeros(g.size, dtype=np.int32)
        last_access[-1] = -1
        for (u, v) in g.edge_list:
            last_access[u] = max(v, last_access[u])

        # Construct R and S matrix for scheduling
        s = gen_s_matrix_fixed_checkpoints(g, g.vfwd)
        # Set 0 after the last access
        for t in range(g.size):
            for i in range(g.size):
                if t > last_access[i]:
                    s[t, i] = 0
        # Set s for eviction
        for e in eviction_set:
            (evicted_access, finish_evicted, back_trigger, back_access) = e.free_time_pair
            s[finish_evicted:back_access, e.tensor_id] = 0
        np.savetxt("S_filled", s, fmt="%i")
        # Set r for recompute
        r = solve_r_opt(g, s)
        np.savetxt("R_filled", r, fmt="%i")

        # generate p and q matrices
        p, q = gen_p_q(s)
        np.savetxt(log_base.joinpath("P-capuchin"), p, fmt="%i")
        np.savetxt(log_base.joinpath("Q-capuchin"), q, fmt="%i")

        schedule, aux_data = schedule_from_rs(g, r, s)
        # log memory timeline
        print(aux_data.mem_timeline)
        schedule_result = ScheduledResult(
            solve_strategy=SolveStrategy.CHECKPOINT_ALL,
            solver_budget=0,
            feasible=True,
            schedule=schedule,
            schedule_aux_data=aux_data,
            solve_time_s=None
        )
        np.savetxt(log_base.joinpath("S"), s, fmt="%i")
        np.savetxt(log_base.joinpath("R"), r, fmt="%i")
        baseline_cpu = np.sum(list(g.cost_cpu.values()))
        peak_mem = aux_data.peak_ram / (1024 * 1024 * 1024)
        compute_overhead = aux_data.cpu * 1.0 / baseline_cpu

        # print(f"Scheduling result for budget {m_device}: peak_mem={peak_mem}, compute overhead={compute_overhead}")
        results.append((peak_mem, compute_overhead, m_device))
    print("Scheduling results:")
    print(results)

