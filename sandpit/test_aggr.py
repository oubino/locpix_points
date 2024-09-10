from torch_geometric.nn import knn_graph, aggr
import torch

mean_aggr = aggr.MeanAggregation()

cluster_id = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
sc_id = torch.tensor([2, 0, 4, 2, 3, 2, 1, 0, 1, 0, 0])
sc_1_id = torch.tensor([2, 0, 1, 2, 0, 2, 0, 0, 0, 0])


superclusterID = mean_aggr(sc_id, index=cluster_id, dim=0).to(torch.int64)

print(superclusterID)

out = mean_aggr(sc_1_id, index=superclusterID, dim=0).to(torch.int64)

print(out)