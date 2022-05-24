from typing import List
import numpy as np
from stropt.core.dfgraph import DFGraph
# import pymetis


# def graph_partition(g: DFGraph, mem_budget: int):
#     adjacent = sorted(g.args.items(), key=lambda x: x[0])
#     print(adjacent)
#     partition = pymetis.part_graph(nparts=3, adjacency=[v[1] for v in adjacent])
#     print(partition)