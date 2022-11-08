import numpy as np
np.set_printoptions(threshold=np.inf)

# Simulate and evaluate the PCIe bandwidth condition
class SwapControler:
    def __init__(self, bd: int, costs: list, feature_map_sizes: list, pq2r=None) -> None:
        self.bandwidth = bd # bandwidth in bytes/second

        # assert len(costs) == len(feature_map_sizes), \
        #                     "List of costs and feature map must have the same length."
        self.nodes_compute_costs = costs
        self.feature_sizes = feature_map_sizes
        self.num_nodes = len(self.feature_sizes)
        # set if use mini-stages
        self.pq2r = pq2r
        if pq2r is not None:
            self.num_stages = len(pq2r.keys())
        else:
            self.num_stages = len(costs.keys())
        self.swap_finish_forward = np.zeros((self.num_nodes, self.num_stages), dtype=np.int16) # (tensor_index, time_stage)
        self.__init_swap_finish_point()
    
    def node_compute_cost(self, index: int):
        if self.pq2r is not None:
            index = self.pq2r[index]
        return self.nodes_compute_costs[index]

    def node_swap_cost(self, index: int):
        if self.pq2r is not None:
            index = self.pq2r[index]
        return int(self.feature_sizes[index] / self.bandwidth * 10E3) # return in ms

    def __swap_point(self, cur_stage: int, tensor_id: int):
        """Find the swap out/in finish point at current stage cur_stage."""
        compute_t_counting = self.node_compute_cost(cur_stage)
        while compute_t_counting < self.node_swap_cost(tensor_id):
            cur_stage += 1
            if cur_stage >= self.num_nodes:
                return -1 # cannot finish swapping in time
            compute_t_counting += self.node_compute_cost(cur_stage)
        return cur_stage
    
    def __init_swap_finish_point(self):
        for t in range(self.num_stages):
            for i in range(t if self.pq2r is None else self.pq2r[t]):
                fp = self.__swap_point(t, i)
                self.swap_finish_forward[i][t] = fp

    def swap_start_stage(self, finish_stage: int, tensor_id: int):
        ''' when to start swap tensor i to finish at stage finish_stage.'''
        if finish_stage <= 0: return None        
        for k in range(self.num_nodes - 1, -1, -1):
            # if there are multiple start points will finish at the same 
            # finish_stage, a larger "k" means less time of bandwidth is occupied.
            if self.swap_finish_forward[tensor_id][k] == finish_stage:
                return k
            # cannot find from matrix, traceback from fp node
            if k == 0:
                return None
            # # This will lead to mismatching when calculating by swap_finish_stage
            # if k == 0:
            #     cur_stage = finish_stage
            #     compute_t_counting = self.node_compute_cost(finish_stage)
            #     while compute_t_counting < self.node_swap_cost(tensor_id):
            #         cur_stage -= 1
            #         if cur_stage < 0:
            #             return None
            #         compute_t_counting += self.node_compute_cost(cur_stage)
            #     return cur_stage
    
    def swap_finish_stage(self, cur_stage: int , tensor_id: int):
        ''' Given the current stage, calculate when the swapping will finish.'''
        if cur_stage >= self.num_nodes or \
            self.swap_finish_forward[tensor_id][cur_stage] == -1:
            return None
        else:
            return self.swap_finish_forward[tensor_id][cur_stage]

    def swap_candidate(self, prefer_occupied: int):
        """Return tuples (t, i) that swapping will occupied the prefer_occupied stage.
        """
        tups = []
        T = self.num_nodes
        for t in range(prefer_occupied + 1):
            for i in range(t):
                sf = self.swap_finish_stage(t, i)
                if sf is not None and sf >= prefer_occupied:
                    tups.append((t, i))
        return tups

def prun_q_opt(sc: SwapControler, q: np.ndarray, s: np.ndarray):
    """We don't put swapping overhead (namely q) in the target,
    so useless swap-in may appear in q. This function for removing
    swap-in which is not checkpointed at (finish_stage + 1).
    """
    T = q.shape[0]
    new_q = np.zeros((T, T), dtype=np.integer)
    for t in range(T):
        if t < T - 1:
            for i in range(t):
                # if q[t, i] == 1:
                if q[t, i] > 0: # will be float number when approx.
                    finish_stage = sc.swap_finish_stage(t, i)    
                    if (finish_stage is not None and finish_stage >= T - 1) \
                        or (finish_stage is not None and s[finish_stage + 1, i] == 0):
                        new_q[t, i] = 0
                        print(f"Pruned Q[{t},{i}]")
                    else:
                        new_q[t, i] = q[t, i]
        else:
            new_q[t, i] = q[t, i]
    return new_q