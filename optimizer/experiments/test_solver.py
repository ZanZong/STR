import logging
import numpy as np
from stropt.tensorflow2.extraction import dfgraph_from_keras
from stropt.core.solvers.graph_reducer import simplify, recover
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model
from experiments.common.profile.cost_model import CostModel

def test_graph_reducer(g):
    return simplify(g, 64)

def test_recover(new_g, origin_g, handler):
    solved_r = "/home/zongzan/dist_dnn_training/STR/optimizer/data/test/R"
    solved_p = "/home/zongzan/dist_dnn_training/STR/optimizer/data/test/P"
    solved_q = "/home/zongzan/dist_dnn_training/STR/optimizer/data/test/Q"
    solved_s = "/home/zongzan/dist_dnn_training/STR/optimizer/data/test/S"
    r = np.loadtxt(solved_r, delimiter=' ')
    p = np.loadtxt(solved_p, delimiter=' ')
    q = np.loadtxt(solved_q, delimiter=' ')
    s = np.loadtxt(solved_s, delimiter=' ')
    reco_r, reco_p, reco_q = recover(origin_g, r, s, p, q, handler)
    print(reco_r.shape)
    print(np.sum(reco_r))
    print(reco_q)
    

if __name__ == "__main__":
    logger = logging.getLogger("test_graph_reducer")
    logger.setLevel(logging.DEBUG)
    model_name = "ResNet50"
    batch_size = 256
    model = get_keras_model(model_name)
    cost_model = CostModel(model_name, "p32xlarge", "", quantization=5)
    cost_model.fit()
    g = dfgraph_from_keras(model, batch_size=batch_size, cost_model=cost_model,
                           loss_cpu_cost=0, loss_ram_cost=(4 * batch_size))
    new_g, handler = test_graph_reducer(g)
    
    test_recover(new_g, g, handler)