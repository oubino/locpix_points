"""Featanalyse"""

from locpix_points.scripts.featanalyse_final_test import main as main_featanalyse


def main():
    # run preprocess on data
    main_featanalyse(
        [
            "-i",
            ".",
            "-c",
            "./config/featanalyse_manual.yaml",
        ]
    )


if __name__ == "__main__":
    main()
