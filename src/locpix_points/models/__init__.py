"""Models

Bring all models into namespace

model_choice returns initialised model,
so that outside of this init file,
don't have to specify the location
of the model in models/ just the name!

Currently available models:

    cluster_net
    loc_cluster_net
    point_net
    point_transformer
"""

from .cluster_nets import ClusterNetHetero, LocClusterNet, ClusterMLP, LocNetClassifyFOV
from .loc_only_nets import LocOnlyNet

# from .simple_gcn_1 import SimpleGCN1
# from .point_net import PointNetEmbedding, PointNetSegmentation
# from .point_transformer import Classifier, Segmenter


def model_choice(name, *args, **kwargs):
    """Returns the chosen model

    Args:
        name (str): Name of the model to initialise
        *args: Positional arguments to initialise the models with
        **kwargs: Keyword arguments to initialise the models with

    Returns:
        model: the model chosen by the user

    Raises:
        ValueError: if desired model is not specified"""

    # if name == "simplegcn1":
    #    return SimpleGCN1(*args)
    # if name == "pointnetembedding":
    #    return PointNetEmbedding(*args)
    # elif name == "pointnetseg":
    #    return PointNetSegmentation(*args)
    # elif name == "pointtransformerseg":
    #    dim = kwargs["dim"]
    #    return Segmenter(*args, dim=dim)
    # elif name == "pointtransformerclass":
    #    dim = kwargs["dim"]
    #    return Classifier(*args, dim=dim)
    if name == "locclusternet":
        device = kwargs["device"]
        return LocClusterNet(*args, device=device)
    elif name == "clusternet":
        return ClusterNetHetero(*args)
    elif name == "clustermlp":
        return ClusterMLP(*args)
    elif name == "locnetonly_pointnet":
        device = kwargs["device"]
        return LocNetClassifyFOV(*args, device=device)
    elif name == "locnetonly_pointtransformer":
        device = kwargs["device"]
        return LocNetClassifyFOV(*args, device=device, transformer=True)
    elif name == "loconlynet":
        device = kwargs["device"]
        return LocOnlyNet(*args, device=device)

    # elif name == "locpointnet":
    #    device = kwargs["device"]
    #    return LocPointNet(*args, device=device)
    else:
        raise ValueError(f"{name} is not a supported model")
