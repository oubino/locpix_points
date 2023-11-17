"""Test pipeline"""

from locpix_points.scripts.process import main

def test_pipeline():
    # run process on data
    main(
        [
            "-i",
            "semore_expts/output",
            "-c",
            "semore_expts/templates/process.yaml",
        ]
    )
