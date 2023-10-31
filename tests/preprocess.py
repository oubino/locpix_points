"""Test pipeline"""

from locpix_points.scripts.preprocess import main as main_preprocess

def test_pipeline():
    # run preprocess on data
    main_preprocess(
        [
            "-i",
            "../../../../mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/code/tma/data/raw/locs",
            "-c",
            "tests/templates/preprocess.yaml",
            "-o",
            "tests/output",
        ]
    )