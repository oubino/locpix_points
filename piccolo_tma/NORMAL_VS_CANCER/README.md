# Normal vs cancer

Label: Cancer = 1; Normal = 0

All 181 FOVs with 38 Normal, 143 Cancer.

## Tasks

| Task ID  | Manual features used | Model | Loc conv type | Cluster conv type |
| ------------- | ------------- | ------------- | ------------- |------------- |
| Task 1  | No  | LocClusterNet | PointNetConv | PointNetConv |
| Task 2  | No  | LocClusterNet | PointTransformer | PointTransformer |