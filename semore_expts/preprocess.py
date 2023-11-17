"""Test pipeline"""

from locpix_points.scripts.preprocess import main 

def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "../../../../mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/semore/data/task_1/train",
            "-c",
            "semore_expts/templates/preprocess.yaml",
            "-o",
            "semore_expts/output",
        ]
    )