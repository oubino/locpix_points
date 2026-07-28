"""Call k-fold split generation (locpix_points/scripts/generate_k_splits.py).

Extra options include:
    -g for grouping data and sharing between
    test and train/validation sets, e.g. for multiple datapoints per patient
    (see help (-h) for the called script).

    -v for verbose output.
"""

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
