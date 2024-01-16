"""Preprocess"""

from locpix_points.scripts.annotate import main as main_annotate
import json


def main():
    # run annotate on data
    main_annotate(
        [
            "-i",
            ".",
            "-c",
            "./config/annotate.yaml",
            "-s",
            "loc",
        ]
    )


if __name__ == "__main__":
    main()
