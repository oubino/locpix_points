"""Test pipeline"""

from locpix_points.scripts.train import main as main_train


def main():
    # run k-fold on data
    main_train(
        [
            "-i",
            ".",
            "-c",
            "./config/train.yaml",
            "-n",
            "final_test",
        ]
    )


if __name__ == "__main__":
    main()
