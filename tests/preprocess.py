"""Test pipeline"""

from locpix_points.scripts.preprocess import main


def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "tests/test_data",
            "-c",
            "tests/config/preprocess.yaml",
            "-o",
            "tests/output",
        ]
    )
