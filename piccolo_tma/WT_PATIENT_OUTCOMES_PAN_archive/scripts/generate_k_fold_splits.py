"""Test pipeline"""

from locpix_points.scripts.generate_k_fold_splits import main as main_gen_k


def main():
    # run k-fold on data
    main_gen_k(
        [
            "-i",
            ".",
            "-c",
            "./config",
            "-s",
            "5",
        ]
    )


if __name__ == "__main__":
    main()
