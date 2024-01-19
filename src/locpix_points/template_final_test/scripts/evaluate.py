"""Test pipeline"""

from locpix_points.scripts.evaluate import main as main_evaluate


def main():
    # run k-fold on data
    main_evaluate(
        [
            "-i",
            ".",
            "-c",
            "./config/evaluate.yaml",
        ]
    )


if __name__ == "__main__":
    main()
