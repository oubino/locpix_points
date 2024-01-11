from torch_geometric.datasets import OSE_GVCS
from torch_geometric.loader import DataLoader

dataset = OSE_GVCS(root="/tmp/OSE_GVCS")
print(len(dataset))

if 1:
    dataset = OSE_GVCS(root="/tmp/OSE_GVCS")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in loader:
        # print(batch)
        print(batch["machine"].batch)

    print(len(dataset))

    dataset = OSE_GVCS(root="/tmp/OSE_GVCS")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        # print(batch)
        print(batch)


# -----------------------
if 0:
    dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES", use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        # print(batch)
        print(batch.x.shape)

    dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES", use_node_attr=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("------------")
    for batch in loader:
        # print(batch)
        print(batch.x.shape)
