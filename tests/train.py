"""Test pipeline"""

from locpix_points.scripts.train import main

def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "tests/output",
            "-c",
            "tests/templates/train.yaml",
        ]
    )
