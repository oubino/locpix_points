"""Preprocess"""

from locpix_points.scripts.preprocess import main as main_pre
import json


def main():
    # run preprocess on data
    metadata_path = "metadata.json"
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        data_path = metadata["data_path"]

    main_pre(
        [
            "-i",
            f"{data_path}/train",
            "-c",
            "./config/preprocess.yaml",
            "-o",
            ".",
            "-f",
            "preprocessed/train",
        ]
    )


if __name__ == "__main__":
    main()
