"""Test pipeline"""

from locpix_points.scripts.process import main as main_process


def main():
    # process
    main_process(
        [
            "-i",
            "semore_expts/task_2",
            "-c",
            "semore_expts/task_2/config/process.yaml",
            "-o",
            f"temp_vis/",
            "-k",
            ["fib_0", "iso_0"],
            "-k",
            [],
            "-k",
            [],
        ]
    )


if __name__ == "__main__":
    main()
