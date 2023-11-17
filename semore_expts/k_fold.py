"""Test pipeline"""

from locpix_points.scripts.k_fold import main

def test_pipeline():
    # run k-fold on data
    main(
        [
            "-i",
            "semore_expts/output",
            "-c",
            "semore_expts/templates/k_fold.yaml",
            "-r",
        ]
    )
