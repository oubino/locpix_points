"""Test pipeline"""

from locpix_points.scripts.featextract import main


def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "tests/output",
            "-c",
            "tests/templates/featextract.yaml",
        ]
    )
