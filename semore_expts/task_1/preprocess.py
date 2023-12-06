"""Test pipeline"""

from locpix_points.scripts.preprocess import main as main_pre


def main():
    # run preprocess on data
    main_pre(
        [
            "-i",
            "../../../../mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/semore/data/dataset_1/train",
            "-c",
            "semore_expts/task_1/config/preprocess.yaml",
            "-o",
            "semore_expts/task_1",
        ]
    )


if __name__ == "__main__":
    main()
