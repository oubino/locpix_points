"""LocClusterNet

Network embeds localisations within clusters.
Concatenates this embedding to user derived cluster features.
GraphNet operates on clusters, goes through linear layer, prediction!

Helped using https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py

"""


# a. Define a GraphNet/PointNet/etc.? that operates on each cluster

# b. Define a GraphNet/etc. that oeprates on the clusters

# c. Define a linear layer that operates on the output of the clusters net

# Order

# 1. Embed each cluster using a.

# 2. Use gra

class LocEmbedder():

    def forward(self, x_dict, edge_index_dict):

        loc_x = self.conv(
            x_dict['loc',
                   edge_index_dict['loc', 'clusteredwith', 'loc']]
        )

        # relu?

        # what shape is loc_x

        # then max_pool or max_pool_neighbour

        # what shape is this

class ClusterEmbedder(MessageParsing):

    def __init__(self):
        super().__init__(aggr='max')
        self.conv = conv.SimpleConv(aggr='sum')

    def forward(self, x_dict, z_dict, edge_index_dict):

        cluster_x = self.conv(
            x_dict['loc','cluster'], edge_index_dict['loc', 'in', 'cluster']
        )

        # concat cluster_x with x_dict['cluster]

        # return    


class LocClusterNet:

    def __init__(self, name):
        self.name = name
        self.loc_embed = 

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        
        z_dict = {}

        # embed each cluster, finish with pooling step
        x_dict['loc'] = self.loc_embed(x_dict)

        # for each cluster concatenate this embedding with previous state 
        x_dict['cluster'] = self.cluster_embed(x_dict)

        # operate graph net on clusters, finish with pooling step
        

        # linear layer


