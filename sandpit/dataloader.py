from torch_geometric.datasets import TUDataset, IMDB
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)

dataset = IMDB(root='/tmp/IMDB')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)
