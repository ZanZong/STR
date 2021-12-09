# Reimplementation of paper "Capuchin: Tensor-based GPU Memory Management for Deep Learning"

from typing import List
import numpy as np

from enum import Enum
from remat.core.dfgraph import DFGraph
from remat.core.utils.swapping import SwapControler

class CAPUCHIN_STATUS(Enum):
    IN = 0
    SWAPPING_OUT = 1
    OUT = 2
    SWAPPING_IN = 3
    RECOMPUTE = 4

# Selected for eviction with swapping
eviction_set = []
# Selected for recomputing
recomps = []

class Tensor:
    def __init__(self, id: str, size: int, access_count: int, srcs: List[int]) -> None:
        self.tensor_id = id
        self.size = size
        self.access_count = access_count
        self.timestamp = None
        self.status = CAPUCHIN_STATUS.IN
        
        # for recomputation
        self.srcs = srcs
        # as a candidate
        self.free_time = None # quantify the swapping
        self.free_time_pair = None
        self.rp_time = None
        self.ext_time = None
        self.MSPS = None # quantify the recompute
    
    def __str__(self) -> str:
        return f"tensor_id:{self.tensor_id}, access_count:{self.access_count}, " + \
            f"status:{self.status}, rp_time:{self.rp_time}, ext_time:{self.ext_time}, MSPS:{self.MSPS}, free_time_pair:{self.free_time_pair}"

def init_msps(g: DFGraph, candidate_set: List[Tensor]):
    for c in candidate_set:
        c.MSPS = g.cost_ram[c.tensor_id] / g.cost_cpu[c.tensor_id]
        c.rp_time = g.cost_cpu[c.tensor_id]

def current_evicted_memory(eviction_set: List[Tensor]):
    return sum([e.size for e in eviction_set])

def hybrid_policy(g: DFGraph, model_name: str, insufficient_memory_budget: int, candidate_set: List[Tensor]):
    swap_control = SwapControler(10E9, g.cost_cpu, g.cost_ram)
    print(f"Candidate set size:{len(candidate_set)}")
    # Step1: determine the eviction set (using swap)
    for c in candidate_set:
        require_c = [(u, v) for (u, v) in g.edge_list if u == c.tensor_id]
        assert len(require_c) >= 2, f"Access time < 2, cannot use {c.tensor_id} as a candidate."
        evicted_access = None
        back_access = None
        finish_evicted = None
        back_trigger = None
        
        evicted_access = require_c[0][1]
        back_access  = require_c[1][1]

        access_c = min([e[1] for e in require_c]) # Capuchin only consider once swap-in
        # more than one nodes depend on c, only use the one with largest FT, (c.tensor_id -> access_c)
        swap_out_end_time = swap_control.swap_finish_stage(evicted_access + 1, c.tensor_id)
        swap_in_start_time = swap_control.swap_start_stage(back_access - 1, c.tensor_id) # finish swap-in before v stage

        if swap_in_start_time is None:
            # find an earler point
            swap_in_finish = access_c - 1
            cur_stage = swap_in_finish
            compute_t_counting = swap_control.node_compute_cost(swap_in_finish)
            trace_back = True
            while compute_t_counting < swap_control.node_swap_cost(c.tensor_id):
                cur_stage -= 1
                if cur_stage < 0:
                    trace_back = False
                    break
                compute_t_counting += swap_control.node_compute_cost(cur_stage)
            if not trace_back:
                continue
            swap_in_start_time = cur_stage
        if swap_in_start_time <= swap_out_end_time:
            continue
        finish_evicted = swap_out_end_time
        back_trigger = swap_in_start_time
        c.free_time = swap_in_start_time - swap_out_end_time
        c.free_time_pair = (evicted_access, finish_evicted, back_trigger, back_access)

    candidate_can_swap = list(filter(lambda c: c.free_time is not None, candidate_set))
    sorted_candis = sorted(candidate_can_swap, key=lambda a: a.free_time, reverse=True)
    # Select tensors from sorted candidate set to cover the insufficent memory budget
    bandwidth_usage = list()
    swapping_only = False
    for candi in sorted_candis:
        bandwidth_allow_swapping = True
        # swapping period need empty bandwidth
        (evicted_access, finish_evicted, back_trigger, back_access) = candi.free_time_pair
        occupied_stages = list(range(evicted_access + 1, finish_evicted + 1))
        occupied_stages.extend(list(range(back_trigger, back_access)))
        if len(occupied_stages) != len(set(occupied_stages)):
            bandwidth_allow_swapping = False
        else:
            for swap_stage in occupied_stages:
                if swap_stage in bandwidth_usage:
                    bandwidth_allow_swapping = False
        if not bandwidth_allow_swapping:
            print(f"{candi} cannot be used because bandwidth")
            continue
        bandwidth_usage.extend(occupied_stages)
        eviction_set.append(candi)
        candidate_set.remove(candi)
        if current_evicted_memory(eviction_set) >= insufficient_memory_budget:
            swapping_only = True
            break
        
    if swapping_only:
        print(f"{len(eviction_set)} swapping covers all insufficent memory budget")
        return eviction_set, [], True
    for evicted in eviction_set:
        evicted.status = CAPUCHIN_STATUS.OUT
    
    print(f"Selected swapping eviction set length={eviction_set}:")
    for e in eviction_set: print(e)
    
    # Step2: cannot satisfy the memory limit, try to use recomputation to expand eviction set.
    # Calculate msps according to the existence of the source tensor, i.e., algorithm2 in the paper.
    evited_mem_size = current_evicted_memory(eviction_set)
    still_insufficient = insufficient_memory_budget - evited_mem_size
    print(f"Swapping covers {evited_mem_size} memory of {insufficient_memory_budget}, recomputation is required")

    feasible = True
    init_msps(g, candidate_set)
    while still_insufficient > 0:
        if len(candidate_set) == 0:
            feasible = False
            print(f"Infeasible solution, still needs {still_insufficient} memory")
            break
        t = max(candidate_set, key=lambda item: item.MSPS)
        ext_ct = 1
        for rp in recomps:
            if t.tensor_id in rp.srcs:
                rp.srcs.remove(t.tensor_id)
                rp.srcs.extend(t.srcs)
                ext_ct += 1
        recomps.append(t)
        candidate_set.remove(t)
        still_insufficient -= t.size
        # Update candidates' MSPS
        for c in candidate_set:
            if t in c.srcs:
                c.srcs.remove(t.tensor_id)
                c.srcs.extend(t.srcs)
                c.rp_time += t.rp_time
                c.ext_time = 0
                for rp in recomps:
                    if c.tensor_id in rp.srcs:
                        c.ext_time += c.rp_time
                if c.tensor_id in t.srcs:
                    c.ext_time = ext_ct * c.rp_time

    return eviction_set, recomps, feasible
