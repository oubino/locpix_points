"""Test pipeline"""

from locpix_points.scripts.featextract import main as main_feat

def main():
    # run preprocess on data
    main_feat(
        [
            "-i",
            "semore_expts/output",
            "-c",
            "semore_expts/templates/featextract.yaml",
        ]
    )

if __name__ == "__main__":
    
    main()