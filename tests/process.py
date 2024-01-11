"""Test pipeline"""

from locpix_points.scripts.process import main


def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "tests/output",
            "-c",
            "tests/config/process.yaml",
        ]
    )
