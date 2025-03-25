"""Test pipeline"""

from locpix_points.scripts.process import main as main_process


def main():
    # run k-fold on data
    main_process(
        [
            "-i",
            ".",
            "-c",
            "./config/process.yaml",
            "-f",
            "preprocessed/train",
            "-f",
            "preprocessed/test",
        ]
    )


if __name__ == "__main__":
    main()
