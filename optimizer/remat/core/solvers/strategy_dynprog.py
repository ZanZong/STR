from typing import List
import numpy as np

from enum import Enum
from remat.core.dfgraph import DFGraph
from remat.core.utils.swapping import SwapControler

def dynprog_policy(g: DFGraph, model_name: str, memory_budget: int, bandwidth: int):
    def fwd_to_bwd(fwd_id):
        return 2 * len(g.v) - fwd_id

    assert len(g.cost_ram_parameters) == len(g.v), "g.cost_ram_parameters {}".format(g.cost_ram_parameters)
    
    # swap_control = SwapControler(10E9, g.cost_cpu, g.cost_ram)

    # DP dict, {(m_f, delta_f, delta_b): idle_time}
    idle = [dict() for i in range(g.size)]
    
    # initial the first layer
    idle[0][(g.cost_ram[0], 0, 0)] = 0
    
    for i in range(g.size - 1):
        print("Processing layer {}".format(i))
        for (m_fi, delta_fi, delta_bi) in idle[i].keys():
            # calc. m_bi for layer i
            m_bi = g.cost_ram[fwd_to_bwd(i)] + max(0, delta_bi) + m_fi - max(0, delta_fi)
            
            cur_mem = max(m_fi + g.cost_ram_parameters[i] - delta_fi, \
                            m_bi + g.cost_ram_parameters[i] + g.cost_ram[i + 1] - delta_bi)
            print("Current memory cost {}".format(cur_mem))
            if g.cost_ram[i + 1] + cur_mem <= memory_budget:
                # incremental memory of forward and backward
                epsilon_fi = max(0, (m_fi + \
                    g.cost_ram_parameters[i] + g.cost_ram[i] - memory_budget) / bandwidth)
                epsilon_bi = max(0, \
                    (m_bi + g.cost_ram_parameters[i] + g.cost_ram[i + 1] + \
                    g.cost_ram[fwd_to_bwd(i)] - memory_budget))
                print("valid, calc. new epsilon_fi, dpsilon_bi: {}, {}".format(epsilon_fi, epsilon_bi))

                # 1. update idle with incremental memory if xi is offloaded
                delta_fi_n = max(0, delta_fi) + g.cost_ram[i] - (epsilon_fi + g.cost_cpu[i]) * bandwidth
                m_fi_n = m_fi + g.cost_ram[i + 1] - \
                    min(max(0, delta_fi) + g.cost_ram[i], (epsilon_fi + g.cost_cpu[i]) * bandwidth)
                delta_bi_n = g.cost_ram[i] + max(0, delta_bi - (epsilon_bi + g.cost_cpu[fwd_to_bwd(i)]) * bandwidth)
                idle[i + 1][(m_fi_n, delta_fi_n, delta_bi_n)] = idle[i][(m_fi, delta_fi, delta_bi)] + epsilon_fi + epsilon_bi
                print("Update tuples for layer {}: {}".format(i + 1, idle[i + 1].keys()))

                # 2. update idle with incremental memory if xi is not offloaded
                delta_fi_n = delta_fi - (epsilon_fi + g.cost_cpu[i]) * bandwidth
                m_fi_n = m_fi + g.cost_ram[i+1] - min(min(0, delta_fi), (epsilon_fi + g.cost_cpu[i]) * bandwidth)
                delta_bi_n = delta_bi - (epsilon_bi + g.cost_cpu[fwd_to_bwd(i)]) * bandwidth
                idle[i + 1][(m_fi_n, delta_fi_n, delta_bi_n)] = idle[i][(m_fi, delta_fi, delta_bi)] + epsilon_fi + epsilon_bi
                print("Update tuples for layer {}: {}".format(i + 1, idle[i + 1].keys()))
    
    print("the last layer solution: {}".format(idle[g.size - 1]))
    total_idle = dict()
    for (m_fi, delta_fi, delta_bi) in idle[-1]:
        epsilon_g = None # TODO
        total_idle[(m_fi, delta_fi, delta_bi)] = idle[-1][(m_fi, delta_fi, delta_bi)] + epsilon_g
    