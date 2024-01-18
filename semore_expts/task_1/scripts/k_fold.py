"""Test pipeline"""

from locpix_points.scripts.k_fold import main as main_k


def main():
    # run k-fold on data
    main_k(
        [
            "-i",
            ".",
            "-c",
            "./config",
        ]
    )


if __name__ == "__main__":
    main()
