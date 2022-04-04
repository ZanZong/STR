import argparse
import numpy as np
import shutil
import pathlib
from stropt.core.dfgraph import DFGraph
from stropt.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from stropt.core.schedule import ScheduledResult
from stropt.core.utils.solver_common import gen_s_matrix_fixed_checkpoints, solve_r_opt
from stropt.core.enum_strategy import SolveStrategy
from stropt.core.utils.scheduler import schedule_from_rs
from stropt.core.utils.timer import Timer
from experiments.common.definitions import remat_data_dir
from experiments.common.load_keras_model import get_keras_model
from experiments.common.profile.cost_model import CostModel
from stropt.tensorflow2.extraction import dfgraph_from_keras
from stropt.core.solvers.strategy_dynprog import dynprog_policy



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


if __name__ == "__main__":
    args = extract_params()
    assert len(args.device_memory) > 0, "At least 1 target device memory is needed."
        
    key = "_".join(map(str, [args.model_name, args.batch_size, args.input_shape]))
    log_base = remat_data_dir() / "dynprog" / key
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
    
    results = []
    for m_device in device_memory:
        target_mem = m_device - g.cost_ram_fixed
        p, q = dynprog_policy(g=g, model=model, memory_budget=m_device, bandwidth=10E9)
        
        np.savetxt(log_base.joinpath("P-dynprog"), p, fmt="%i")
        np.savetxt(log_base.joinpath("Q-dynprog"), q, fmt="%i")

        
    

