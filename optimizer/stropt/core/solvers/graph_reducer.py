import copy, logging
from re import L
import numpy as np
from typing import Tuple
from stropt.core.dfgraph import DFGraph, EdgeList, Vertex

logger = logging.getLogger("GraphReducer")

class FuseHandler:
    """Do the fusion operations. Because the use of swapping is far more than recomputation
    in our approach, we prefer to fuse the nodes in DAG graph to reduce swapping overheads.
    For example, the edge between vertexs is weighted with swapping overheads, and fused 
    according to the a greedy policy.

    Save fused edges for the new graph building and recovering.
    """
    def __init__(self, g: DFGraph):
        self.__g = g
        # to save all fusions
        self.__fuse_logger = dict()
        # a node index mapping between the new graph and the previous simplified graph
        self.__newgraph2old = dict()
        self.__old2newgraph = dict()
        self.origin_size = g.size

    def fuse(self, v1: Vertex, v2: Vertex):
        """Fusing vertex v1 and v2 of graph g as a new vertex v, and delete the edge between v1 and v2.
        """
        if not abs(v1 - v2) == 1 and (v1, v2) in self.__g.edge_list:
            logger.warn("Cannot fuse nonadjacent vertexes in topology order")
            return

        # remove the max-index vertex and link its edge to another vertex
        removed_v = max(v1, v2) 
        remained_v = min(v1, v2)
        self.__g.v.remove(removed_v)
        for u, deps in self.__g.args.items():
            if u == removed_v:
                # fix fused node dependency
                for dep in self.__g.args[removed_v]:
                    if dep == remained_v or dep in self.__g.args[remained_v]: continue
                    self.__g.args[remained_v].append(dep)
            else:
                # fix others dependent on removed node
                for i, dep in enumerate(deps):
                    if dep == removed_v:
                        self.__g.args[u][i] = remained_v
        del self.__g.args[removed_v]
        self.__g.cost_cpu[remained_v] += self.__g.cost_cpu[removed_v]
        self.__g.cost_ram[remained_v] += self.__g.cost_ram[removed_v]
    
    def set_ng2pre(self, ng_id, pre_id):
        self.__newgraph2old[ng_id] = pre_id

    def set_pre2ng(self, pre_id, ng_id):
        self.__old2newgraph[pre_id] = ng_id

    def get_fused_tensors(self, final_id):
        """ Get fused tensors at final_id in the original graph.
        """
        if final_id not in self.__fuse_logger.keys():
            return None # final_id not a fused node
        else:
            return self.__fuse_logger[final_id]

    def collect_all_fusion(self):
        """ Constructing the index mapping between old graph and fused graph.
        Need to be invoked after all fusion.
        """
        for i in range(len(self.__g.v) - 1):
            if (self.__g.v[i + 1] - self.__g.v[i]) > 1:
                self.__fuse_logger[self.__g.v[i]] = [i for i in range(self.__g.v[i], self.__g.v[i + 1] + 1)]

    def get_ng_id(self, pre_id):
        if pre_id not in self.__old2newgraph.keys():
            logger.warn(f"Cannot find index in new graph w.r.t old id={pre_id}")
        return self.__old2newgraph[pre_id]
    
    def get_og_id(self, ng_id):
        if ng_id not in self.__newgraph2old.keys():
            logger.warn(f"Cannot find index in old graph w.r.t new graph id={ng_id}")
        return self.__newgraph2old[ng_id]

def simplify(g: DFGraph, target_n) -> Tuple[DFGraph, FuseHandler]:
    """Fusing tensors greedily. Current strategy will try to aggregate small tensors.
    """
    if g.size <= target_n:
        logger.info(f"Current graph nodes {g.size} <= {target_n}, doesn't need simplify.")
        return None
    tmp_g = copy.deepcopy(g)
    fuse_handler = FuseHandler(g=tmp_g)
    tensor_rank = list(np.argsort(tmp_g.cost_ram[i] for i in range(tmp_g.size))[::-1]) # descend order

    while tmp_g.size > target_n:
        assert len(tensor_rank) > 0, "Cannot reach the target number of nodes."
        fused_idx = tensor_rank.pop()
        dest = tmp_g.successors(fused_idx)
        sour = tmp_g.predecessors(fused_idx)
        # find a neighbor with smallest tensor size
        smallest_neighbor = np.inf
        smallest_neighbor_idx = 0
        for i in dest.union(sour):
            if tmp_g.cost_ram[i] < smallest_neighbor:
                smallest_neighbor = tmp_g.cost_ram[i]
                smallest_neighbor_idx = i
        fuse_handler.fuse(fused_idx, smallest_neighbor_idx)
    fuse_handler.collect_all_fusion()
    # buid a new graph w.r.t tmp_g
    depends = dict() # dependency for each node
    vertexs = [i for i in range(len(tmp_g.v))]
    cost_cpu = [tmp_g.cost_cpu[i] for i in tmp_g.v]
    cost_ram = [tmp_g.cost_ram[i] for i in tmp_g.v]
    node_names = [tmp_g.node_names[i] for i in tmp_g.v]
    cost_ram_parameters = g.cost_ram_parameters # the total ram of parameters doesn't change

    # record the map of new graph vertex with the old
    for i, v in enumerate(tmp_g.v):
        fuse_handler.set_ng2pre(i, v)
        fuse_handler.set_pre2ng(v, i)
    # set dependencies for each node
    for (u, deps) in tmp_g.args.items():
        depends[fuse_handler.get_ng_id(u)] = [fuse_handler.get_ng_id(dep) for dep in deps]

    return DFGraph(depends, vertexs, None, tmp_g.vloss, cost_cpu, cost_ram, node_names, cost_ram_parameters, symmetric=False), fuse_handler


def recovery(r: np.ndarray, s: np.ndarray, q: np.ndarray, handler: FuseHandler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recovering a simplified graph to a normal forward-backward symmetric graph.
    """
    T = handler.origin_size
    NT = r.shape[0]
    R_ = np.eye(T, dtype=r.dtype)
    S_ = np.zero((T, T), dtype=s.dtype)
    Q_ = np.zero((T, T), dtype=q.dtype)
    for t in range(NT):
        for i in range(t):
            # whether is fused in the old graph
            fused_i = handler.get_fused_tensors(handler.get_og_id(i))
            fused_t = handler.get_fused_tensors(handler.get_og_id(t))
            if fused_i is None and fused_t is None:
                pass
            # if r[t, i] == 1:
            #     if fused is None:
            #         R_[handler.get_og_id(t), handler.get_og_id(i)] = 1
            #     else:
            #         for tensor in fused:
            #             R_[handler.get_og_id(t), tensor] = 1
            # if q[t, i] == 1:
            #     if fused is None:
            #         Q_[handler.get_og_id(t), handler.get_og_id(i)] = 1
            #     else:
            #         # will result in multiple swap-in operations triggered at the same time (packaged swap)
            #         for tensor in fused:
            #             Q_[handler.get_og_id(t), tensor] = 1
            # if s[t, i] == 1:
            #     # fill the region of fused_t and fused_i
            #     if fused_i is None and fused_t is None:
            #         S_[handler.get_og_id(t), handler.get_og_id(i)] = 1
            #     elif fused_i is None and fused_t is not None:
            #         for ft in fused_t:
            #             S_[ft, handler.get_og_id(i)] = 1
            #     else:
            #         for tensor in fused:
            #             S_[handler.get_og_id(t), handler.get_og_id(tensor)] = 1
    
    # TODO source tracing to fix missing dependencies

            
                