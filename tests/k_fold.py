"""Test pipeline"""

from locpix_points.scripts.k_fold import main


def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "tests/output",
            "-c",
            "tests/config/",
            "-r",
            "5",
        ]
    )
