"""Test pipeline"""

from locpix_points.scripts.featextract import main

def test_pipeline():
    # run preprocess on data
    main(
        [
            "-i",
            "semore_expts/output",
            "-c",
            "semore_expts/templates/featextract.yaml",
        ]
    )
