"""Test pipeline"""

from locpix_points.scripts.k_fold import main as main_k


def main():
    # run k-fold on data
    main_k(
        [
            "-i",
            "semore_expts/task_1",
            "-c",
            "semore_expts/task_1/config",
        ]
    )


if __name__ == "__main__":
    main()
