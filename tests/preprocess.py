"""Test pipeline"""

from locpix_points.scripts.preprocess import main


def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "../../../../mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/tma_genetech/data/raw/locs",
            "-c",
            "tests/templates/preprocess.yaml",
            "-o",
            "tests/output",
        ]
    )
