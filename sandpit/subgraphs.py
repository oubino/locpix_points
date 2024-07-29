import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [5]], dtype=torch.float)
pos = torch.tensor([[0.1, 1.2], [0.2, 1.3], [1.5, 12.2], [1.5, 4.5]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, pos=pos)

node = False
edge = True

imp_nodes = torch.tensor([False,True,True, False])
non_imp_nodes = torch.where(imp_nodes == False, True, False)
imp_edges = torch.tensor([False,False,True,False, False, False])
non_imp_edges = torch.where(imp_edges == False, True, False)

if node:
    # automatically relabelled
    subgraph = data.subgraph(imp_nodes)
    if subgraph.num_nodes == data.num_nodes:
        raise ValueError("No complement graph as induced subgraph is the whole graph")
    else:  
        complement_graph = data.subgraph(non_imp_nodes)
    print(subgraph.x)
    print(complement_graph.x)
    print(complement_graph.pos)
elif edge:
    nx_graph = to_networkx(data, node_attrs=["x", "pos"])
    include_edges = data.edge_index.T[imp_edges]
    include_edges = include_edges.numpy()
    include_edges = list(map(tuple, include_edges))
    subgraph = nx_graph.edge_subgraph(include_edges)
    subgraph_pyg = from_networkx(subgraph, group_node_attrs=["x", "pos"])
    print("subgraph")
    x = subgraph_pyg.x[:,:-2]
    pos = subgraph_pyg.x[:,-2:]
    subgraph_pyg.x = x
    subgraph_pyg.pos = pos
    print(subgraph_pyg.x)
    print(subgraph_pyg.pos)
    print(subgraph_pyg.edge_index)
    if subgraph.nodes == nx_graph.nodes:
        raise ValueError("No complement graph as induced subgraph is the whole graph")
    else:    
        print("complement")
        e = list(subgraph.nodes)
        nx_graph.remove_nodes_from(e)

        complement_graph_pyg = from_networkx(nx_graph, group_node_attrs=["x", "pos"])
        print(complement_graph_pyg.x)
        print(complement_graph_pyg.pos)
        print(complement_graph_pyg.edge_index)

