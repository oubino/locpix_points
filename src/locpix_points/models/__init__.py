"""Init file

Bring all models into namespace

model_choice returns initialised model,
so that outside of this init file,
don't have to specify the location
of the model in models/ just the name!
"""

from .simple_gcn_1 import SimpleGCN1
from .point_net import PointNetClassification, PointNetSegmentation


def model_choice(name, *args):
    if name == 'simplegcn1':
        return SimpleGCN1(*args)
    elif name == 'pointnetclass':
        return PointNetClassification(*args)
    elif name == 'pointnetseg':
        return PointNetSegmentation(*args)
    else:
        raise ValueError(f'{name} is not a supported model')
