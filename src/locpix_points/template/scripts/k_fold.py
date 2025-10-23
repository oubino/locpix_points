"""Test pipeline"""

from locpix_points.scripts.k_fold import main as main_k
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="k-fold")

    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="starting fold",
        required=False,
    )

    args = parser.parse_args(argv)

    if args.fold is None:
        # run k-fold on data
        main_k(
            [
                "-i",
                ".",
                "-c",
                "./config",
            ]
        )

    else:
        # run k-fold on data
        main_k(
            [
                "-i",
                ".",
                "-c",
                "./config",
                "-f",
                f"{args.fold}",
            ]
        )


if __name__ == "__main__":
    main()
