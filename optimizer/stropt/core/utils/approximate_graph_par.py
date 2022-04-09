from typing import List
import numpy as np
from stropt.core.dfgraph import DFGraph
import pymetis


def graph_partition(g: DFGraph, mem_budget: int):
    adjacent = sorted(g.args.items(), key=lambda x: x[0])
    print(adjacent)
    partition = pymetis.part_graph(nparts=3, adjacency=[v[1] for v in adjacent])
    print(partition)
    render_graph(g, adjacent, partition[1], "/home/zongzan/dist_dnn_training/STR/optimizer", "vgg16")


def render_graph(g: DFGraph, adj_list: List, partition: List, directory, name=""):
    """Generate Graphviz-formatted edge list for visualization, and write pdf

    Args:
        g (DFGraph): _description_
        adj_list (List): 
        partition (List): list of partition number of 
        directory (_type_): pdf file path
        name (str, optional): model name. Defaults to "".
    """    
    from graphviz import Digraph

    colors = ['black', 'darkred', 'green', 'yellow', 'blue', \
                'orangered', 'grey', 'darkviolet', 'saddlebrown', 'steelblue']
    get_color = lambda index: colors[index % len(colors)]
    if max(partition) >= len(colors):
        print("Warning: color list is out of range, repeat use.")

    dot = Digraph("graph_" + str(name))
    dot.attr('graph', ratio='compress')
    partition.insert(0, partition[0]) # `input` node in the first partition
    
    for u in g.vfwd:
        with dot.subgraph() as s:
            s.attr(rank='same')
            node_name = g.node_names.get(u)
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
            s.node(str(u), node_name, color=get_color(partition[u]))

            v = g.forward_to_backward(u)
            node_name = "&nabla;{}".format(g.node_names.get(u, u))
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(v))
            s.node(str(v), node_name, color=get_color(partition[v]))

    for u in g.v:
        if u not in g.vfwd_map.values() and u not in g.vfwd_map.keys():
            node_name = g.node_names.get(u)
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
            dot.node(str(u), node_name, color=get_color(partition[u]))

    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0]))
        if edge not in g.edge_list_fwd and g.vloss not in edge:
            dot.edge(*map(str, edge), constraint='false', label=dep_order)
        else:
            dot.edge(*map(str, edge), label=dep_order)
    try:
        dot.render(directory=directory, format='pdf', quiet=True)
    except TypeError:
        dot.render(directory=directory, format='pdf')