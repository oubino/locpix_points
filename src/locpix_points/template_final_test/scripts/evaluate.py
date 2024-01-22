"Evaluation"

from locpix_points.scripts.evaluate import main as main_evaluate
import os

def main():
    model_list = os.listdir("./models")
    assert len(model_list) == 0 
    model_loc = os.path.join("./models", model_list[0])
    main_evaluate(
        [
            "-i",
            ".",
            "-c",
            "./config/evaluate.yaml",
            "-m",
            f"{model_loc}",
            "-n",
            "final_test",
        ]
    )


if __name__ == "__main__":
    main()
