"""Test pipeline"""

from locpix_points.scripts.featextract import main as main_feat

def main():
    # run preprocess on data
    main_feat(
        [
            "-i",
            "semore_expts/task_1",
            "-c",
            "semore_expts/task_1/config/featextract.yaml",
        ]
    )

if __name__ == "__main__":
    
    main()