"""Test pipeline"""

from locpix_points.scripts.preprocess import main as main_pre 

def main():
    # run preprocess on data
    main_pre(
        [
            "-i",
            "../../../../mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/semore/data/task_1/train",
            "-c",
            "semore_expts/templates/preprocess.yaml",
            "-o",
            "semore_expts/output",
        ]
    )

if __name__ == "__main__":
    main()