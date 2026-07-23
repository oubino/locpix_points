# Rename NN models

import os

for fold in range(5):

    fold_folder = f"output/models/fold_{fold}"

    models = os.listdir(fold_folder)
    
    assert len(models) == 1

    model = models[0]

    model_pth = os.path.join(
        fold_folder, model
    )

    new_model_pth = os.path.join(
        fold_folder, "nn_model.pt"
    )

    os.rename(model_pth, new_model_pth)
