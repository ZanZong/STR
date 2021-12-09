
from remat.core.dfgraph import DFGraph
from remat.core.utils.swapping import SwapControler
import numpy as np

def check_recompute_swap_mutex(sc: SwapControler, R, Q, shape):
    '''Recomputing is not needed at the stage that will swap-in the tensor'''
    for t in range(shape):
        for i in range(shape):
            if R[t, i] == 1:
                start_stage = sc.swap_start_stage(t, i)
                if start_stage is not None:
                    assert Q[start_stage, i] == 0, f"Break the mutex rule of R and Q for t={t}, i={i}."

def check_correctness_of_swap(P, Q, shape):
    for i in range(shape):
        count_p = sum([P[t, i] for t in range(shape)])
        count_q = sum([Q[t, i] for t in range(shape)])
        if count_q > 0:
            assert count_p == 1, f"Break the swapping rule of P and Q, tensor {i} used swap-in without swap-out."

def check_compute_correctness(g: DFGraph, s: np.ndarray, r: np.ndarray):
    T = s.shape[0]
    assert s.shape[1] == T
    # Create reverse adjacency list (child -> parents, i.e. node -> dependencies)
    adj = [[] for _ in range(T)]
    for (u, v) in g.edge_list:
        adj[v].append(u)
    # Enforce R_{t,v} <= R_{t,u} + S_{t,u} for all (u, v) \in E
    correct = True
    for t in range(T):
        for v in range(t, -1, -1):
            for u in adj[v]:
                if r[t, v] > r[t, u] + s[t, u]:
                    correct = False
                    print(f"Break compute rule:r[{t}, {v}] > r[{t}, {u}] + s[{t}, {u}]")
    return correct

if __name__ == "__main__":
    Q = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/Q", delimiter=" ")
    QB = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/QB", delimiter=" ")
    P = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/P", delimiter=" ")
    PB = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/PB", delimiter=" ")
    R = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/R", delimiter=" ")
    S = np.loadtxt("/home/zongzan/dist_dnn_training/MixedMemoryOpt/data/budget_sweep/p32xlarge_VGG16_200_None/ilp_log/S", delimiter=" ")

    T = R.shape[0]
    check_correctness_of_swap(P, Q, T)
    