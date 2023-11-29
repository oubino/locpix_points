"""Test pipeline"""

from locpix_points.scripts.visualise import main as main_vis


def main():
    main_vis(
        [
            "-i",
            "semore_expts/task_2/temp_vis/train/0.pt",
            "-x",
            "x",
            "-y",
            "y",
            "-c",
            "channel",
        ]
    )

if __name__ == "__main__":
    main()