"""Test pipeline"""

from locpix_points.scripts.featextract import main as main_featextract

def test_pipeline():
    # run preprocess on data
    main_featextract(
        [
            "-i",
            "tests/output",
            "-c",
            "tests/templates/featextract.yaml",
        ]
    )
