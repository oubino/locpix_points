import polars as pl
import numpy as np
from torch_geometric import transforms
from locpix_points.data_loading import custom_transforms
import torch

data = torch.load('tests/output/processed/train/0.pt')


torch.save(data, 'sandpit/output/pre_transform.pt')

output_transforms = []

# axis to rotate around i.e. axis=2 rotate around z axis - meaning
# coordinates are rotated in the xy plane
output_transforms.append(custom_transforms.RandomRotate(degrees=180, axis=2))

# need to either define as constant or allow precision to impact this
#output_transforms.append(custom_transforms.RandomJitter(0.1))

# axis = 0 - means x coordinates are flipped - i.e. reflection
# in the y axis
#output_transforms.append(custom_transforms.RandomFlip(axis=0))

# axis = 1 - means y coordinates are flipped - i.e. reflection
# in the x axis
#output_transforms.append(custom_transforms.RandomFlip(axis=1))

# need to define scale factor interval in config
#output_transforms.append(
#    custom_transforms.RandomScale(scales=tuple([0.5, 0.2]))
#)

# shear by particular matrix
#output_transforms.append(custom_transforms.RandomShear(0.3))

#output_transforms.append(
#    custom_transforms.Subsample(
#            transform["subsample"][0], transform["subsample"][1]
#        )
#    )

output_transforms = transforms.Compose(output_transforms)

data = output_transforms(data)

# save data
torch.save(data, 'sandpit/output/post_transform.pt')
