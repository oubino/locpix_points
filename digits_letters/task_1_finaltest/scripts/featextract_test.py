"""Test pipeline"""

from locpix_points.scripts.featextract import main as main_feat


def main():
    # run preprocess on data
    main_feat(
        [
            "-i",
            ".",
            "-c",
            "./config/featextract.yaml",
            "-f",
            "preprocessed/test",
        ]
    )


if __name__ == "__main__":
    main()
