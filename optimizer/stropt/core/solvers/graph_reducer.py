import copy, logging
from functools import reduce
import numpy as np
from typing import List, Tuple
from stropt.core.dfgraph import DFGraph, EdgeList, Vertex
from collections import defaultdict

logger = logging.getLogger("GraphReducer")

class FuseHandler:
    """Do the fusion operations. Because the use of swapping is far more than recomputation
    in our approach, we prefer to fuse the nodes in DAG graph to reduce swapping overheads.
    For example, the edge between vertexs is weighted with swapping overheads, and fused 
    according to the a greedy policy.

    Save fused edges for the new graph building and recovering.
    """
    def __init__(self, g: DFGraph, weight: List):
        self.__g = g
        # to save all fusions
        # self.__fuse_logger = dict()
        # a node index mapping between the simplified new graph and the previous old graph
        self.__fused_in_newgraph = defaultdict(list)
        # mapping from old graph vertex to group in new graph, many-to-one
        self.__v2group = dict()
        self.origin_size = g.size
        self.fused_size = None
        # vertex weight for fusing
        self.weight = weight
        # loss node
        self.v_loss = g.vloss

    def fuse(self, removed_v: Vertex, remained_v: Vertex):
        """Fusing removed_v to remained_v of graph g, and delete the edge between two vertices.
           The removed_v vertex will be set to None, which means the length of node list will not change during fusing.
           Always fuse large-indexed vertex to smaller one.
        """

        assert removed_v is not None and remained_v is not None
        # Fix fused node dependency
        predecessors = list(self.__g.predecessors(removed_v))
        # if remained_v == 41:
        #     print(f"k41's predecessor is {predecessors}")
        for idx, pred in enumerate(predecessors):
            # Check if predecessor has been fused
            if self.__g.args[pred] is None:
                fused_dep = self.find_fused_dep(pred),
                assert fused_dep is not None
                predecessors[idx] = fused_dep

        # Inherit removed predecessor's dependency
        group = []
        for v in predecessors:
            if v != remained_v and v not in self.__g.args[remained_v]:
                if isinstance(v, Tuple):
                    # if v[0] == remained_v: continue
                    group.append(v[0])
                else:
                    # if v == remained_v: continue
                    group.append(v)
        if removed_v in group:
            group.remove(removed_v)
        self.__g.args[remained_v].extend(group)

        # if remained_v == 41:
        #     print(f"k41's predecessor is {predecessors}, change in args: {self.__g.args[remained_v]}")

        successors = list(self.__g.successors(removed_v))
        for idx, succ in enumerate(successors):
            if self.__g.args[succ] is None:
                fused_succ = self.find_fused_dep(succ)
                assert fused_succ is not None
                successors[idx] = fused_succ
        for succ in successors:
            if succ == remained_v: continue
            for idx, v in enumerate(self.__g.args[succ]):
                if v == removed_v:
                    self.__g.args[succ][idx] = remained_v
        
        if removed_v == self.v_loss:
            self.v_loss = remained_v
        # remove edges, node and weight
        self.__g.args[removed_v] = None
        self.__g.v[removed_v] = None
        # __newgraph2old[2]:[3,4] means nodes 2,3,4 are fused to one node
        if removed_v in self.__fused_in_newgraph.keys():
            self.__fused_in_newgraph[remained_v].extend(self.__fused_in_newgraph[removed_v])
            del self.__fused_in_newgraph[removed_v]
        self.__fused_in_newgraph[remained_v].append(removed_v)
        self.weight[remained_v] += self.weight[removed_v]
        self.weight[removed_v] = None


    def collect_graph(self):
        """ Greate a new graph according to fusing. The index mapping between new graph and old graph 
            is recorded in self.__node_mapping.
        """
        depends = dict() # dependency for each node
        vertex = [v for v in range(self.get_graph_size())]
        
        # Remove dumplication
        for v, deps in self.__g.args.items():
            if deps is None: continue
            self.__g.args[v] = list(set(deps))
        print(f"before collect graph {self.__g.args}")
        print(f"Fusing results __fused_in_newgraph: {self.__fused_in_newgraph}")

        # 1. Construct depends using find successor of each group of __fused_in_newgraph, remove vertex in this group
        group_deps = dict()
        for remain_v, fused_v in self.__fused_in_newgraph.items():
            group = [remain_v] + fused_v
            group_predecessors = set()
            for v in group:
                group_predecessors = group_predecessors.union(self.__g.predecessors(v))
            group_predecessors.difference_update(set(group))
            group_deps[remain_v] = list(group_predecessors)
        print(f"Group dependency: {group_deps}")
        
        # 2. Shrink args' key first, get group index mapping
        cur_node = 0
        for v in self.__g.v:
            # Fuse to none
            if v is None:
                continue
            self.__v2group[v] = cur_node
            cur_node += 1
        for key, values in self.__fused_in_newgraph.items():
            for value in values:
                self.__v2group[value] = self.__v2group[key]
        assert len(self.__v2group.keys()) == self.origin_size, "Missing node for __v2group"
        print(f"self.__v2group: {self.__v2group}")

        # 3. Fix depends value by __v2group
        for node, deps in self.__g.args.items():
            if deps is None:
                continue
            if node in group_deps.keys():
                # be fused but remained
                depends[self.__v2group[node]] = list(set([self.__v2group[predecessor] for predecessor in group_deps[node]]))
            else:
                # not fused
                depends[self.__v2group[node]] = list(set([self.__v2group[unfused_node] for unfused_node in deps]))
            if len(depends[self.__v2group[node]]) <= 0:
                del depends[self.__v2group[node]]
        print(f"New args after mapping: {depends}")

        self.fused_size = len(vertex)
        print(f"Mapping original graph N={self.origin_size} to N={self.fused_size}")

        def fused_cost(costs):
            cost_of_newgraph = dict()
            for key in self.__v2group.keys():
                new_k = self.__v2group[key]
                if new_k not in cost_of_newgraph.keys():
                    cost_of_newgraph[new_k] = costs[key]
                else:
                    cost_of_newgraph[new_k] += costs[key]
            return cost_of_newgraph

        cost_cpu = fused_cost(self.__g.cost_cpu)
        cost_ram = fused_cost(self.__g.cost_ram)
        node_names = defaultdict(list)
        for key in self.__v2group.keys():
            new_k = self.__v2group[key]
            if key in self.__g.node_names.keys():
                node_names[new_k].append(self.__g.node_names[key])
        for key in node_names.keys():
            node_names[key] = ",".join(node_names[key])
        cost_ram_parameters = self.__g.cost_ram_parameters
        print(f"Loss node in old graph={self.__g.vloss}, in new graph={self.__v2group[self.v_loss]}")
        # print(f"Create new graph after fusing: {self.__g.v}")
        reduced_graph = DFGraph(depends, vertex, None, self.v_loss, cost_cpu, cost_ram, node_names, cost_ram_parameters, symmetric=False)
        return reduced_graph
        
    def find_fused_dep(self, dep):
        k = None
        for key in self.__fused_in_newgraph.keys():
            nodes = self.__fused_in_newgraph[key]
            for v in nodes:
                if v == dep:
                    k = key
        if k is None:
            print(f"Cannot find {dep} in fused nodes")
        return k

    def get_graph_size(self) -> int:
        """Return current un-none vertex in self.__g

        Returns:
            int: graph size
        """
        count = 0
        for v in self.__g.v:
            if v is not None:
                count += 1
        return count

    def tensor_selection(self):
        """Return tensors that need to be fused.
           Heuristics:1) start from vertex with minimal weight, and
                      2) must fusing the vertex among topology-ordered lower-weighted adjacent dependencies.
        """
        v_f = np.argmin([w if w is not None else np.inf for w in self.weight])
        assert v_f is not None, "All vertex is None!"
        
        def step(v_f: Vertex, direct: str):
            """ Find un-none node with direct 'pred' and 'succ'
            """ 
            cur = v_f
            if direct == 'pred':
                cur -= 1
            else:
                cur += 1
            if cur < 0 or cur >= self.origin_size:
                return None
            while self.__g.v[cur] is None:
                if direct == 'pred':
                    cur -= 1
                else:
                    cur += 1
                if cur < 0 or cur >= self.origin_size:
                    return None
            return cur

        adj_node = [step(v_f, 'pred'), step(v_f, 'succ')]
        assert any(adj_node), f"Failed to find valid node for {v_f} among {self.__g.v}"
        min_weight = np.inf
        min_v = None

        for v in adj_node:
            if v is None: continue
            if self.weight[v] < min_weight:
                min_weight = self.weight[v]
                min_v = v
        # print(f"Selected tensor {v_f} with weight {self.weight[v_f]}, fuse {v_f} to neighbor {min_v}")
        return v_f, min_v
            
    def get_fused_nodes(self):
        fused = list()
        for key in self.__fused_in_newgraph.keys():
            fused.append(self.__v2group[key])
        return fused

    def get_v2group(self):
        return self.__v2group

def simplify(g: DFGraph, target_n: int) -> Tuple[DFGraph, FuseHandler]:
    """Fusing tensors with some heuristics.

    Args:
        g (DFGraph): origional graph
        target_n (int): the number of graph nodes after simplifing.

    Returns:
        Tuple[DFGraph, FuseHandler]: new graph and FuseHandler instance for recovering.
    """    
    if g.size <= target_n:
        logger.info(f"Current graph nodes {g.size} <= {target_n}, doesn't need simplify.")
        return None
    tmp_g = copy.deepcopy(g)
    
    print(f"Graph size={tmp_g.size}")
    # edge-weighted rank
    edge_count = [len(deps) if len(deps) != 0 else 1 for u, deps in tmp_g.successor_dict.items()]
    edge_count.append(1) # for last node without successor
    print(f"--- Edge count (len={len(edge_count)}):")
    # print(edge_count)
    edge_weighted_cost = [tmp_g.cost_ram[i] * edge_count[i] for i in range(tmp_g.size)]
    # Set loss node to inf to avoid node fusing
    edge_weighted_cost[g.vloss] = np.inf
    fuse_handler = FuseHandler(g=tmp_g, weight=edge_weighted_cost)

    # Node fusing
    while fuse_handler.get_graph_size() > target_n:
        remove, remain = fuse_handler.tensor_selection()
        fuse_handler.fuse(remove, remain)
    
    # Greating new graph for optimization
    new_graph = fuse_handler.collect_graph()
    # print(new_graph.node_names)
    # Plot graph
    # render_graph(new_graph, fuse_handler.get_fused_nodes(), "/home/zongzan/dist_dnn_training/STR/optimizer/experiments", "cgraph")

    return (new_graph, fuse_handler)


def recover(origin_g: DFGraph, r: np.ndarray, s: np.ndarray, p: np.ndarray, q: np.ndarray, handler: FuseHandler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recovering a simplified graph to a normal forward-backward symmetric graph.
    """
    T = handler.origin_size
    NT = r.shape[0]
    R_ = np.eye(T, dtype=r.dtype)
    S_ = np.zeros((T, T), dtype=s.dtype)
    P_ = np.zeros((T, T), dtype=p.dtype)
    Q_ = np.zeros((T, T), dtype=q.dtype)
    
    grouped = defaultdict(list)
    for origin_v, new_v in handler.get_v2group().items():
        grouped[new_v].append(origin_v)

    for t in range(NT):
        for i in range(t):
            mapped_nodes = grouped[i]
            if p[t, i] == 1:
                # start swap out after generation
                for node in mapped_nodes:
                    P_[node + 1, node] = 1
            if q[t, i] == 1:
                # start swap in simultaneously at grouped[t] stage
                first_stage_in_group = min(grouped[t])
                for node in mapped_nodes:
                    Q_[first_stage_in_group, node] = 1
            if s[t, i] == 1:
                for node in mapped_nodes:
                    for stage in grouped[t]:
                        S_[stage, node] = 1
    # Fix R
    # sdiff = S_[1:] - S_[:-1]
    # R_[:-1] = R_[:-1] | (R_[:-1] < sdiff)
    # adj = [[] for _ in range(T)]
    # for (u, v) in origin_g.edge_list:
    #     adj[v].append(u)
    # # Swapping has been reflected on s
    # for t in range(T):
    #     for v in range(t, -1, -1):
    #         for u in adj[v]:
    #             if R_[t, v] > R_[t, u] + S_[t, u]:
    #                 R_[t, u] = 1
    return R_, P_, Q_

def render_graph(g: DFGraph, fused: List, directory, name=""):
    """Generate Graphviz-formatted edge list for visualization, and write pdf

    Args:
        g (DFGraph): _description_
        fused (List): index of fused nodes )
        directory (_type_): pdf file path
        name (str, optional): model name. Defaults to "".
    """    
    from graphviz import Digraph

    colors = ['black', 'darkred', 'green', 'yellow', 'blue', \
                'orangered', 'grey', 'darkviolet', 'saddlebrown', 'steelblue']
    get_color = lambda index: colors[index % len(colors)]
    
    dot = Digraph("graph_" + str(name))
    dot.attr('graph', ratio='compress')

    for u in g.v:
        node_name = g.node_names[u]
        node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
        if u in fused:
            dot.node(str(u), node_name, color='blue')
        else:
            dot.node(str(u), node_name, color='black')

    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0])) 
        dot.edge(*map(str, edge), label=dep_order)
    try:
        dot.render(directory=directory, format='pdf', quiet=True)
    except TypeError:
        dot.render(directory=directory, format='pdf')
