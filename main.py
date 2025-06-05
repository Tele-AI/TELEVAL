import argparse
import logging

from src.task import Pipeline

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Global config which will override the settings in the YAML file")
    parser.add_argument("--mode", default="eval", choices=["infer", "eval"])
    parser.add_argument("--task", default="aqa", help="infer task")
    parser.add_argument("--model", default=None)
    
    parser.add_argument("--bsz", default=None, help="batch size for infer and eval")
    parser.add_argument("--save_dir", default="res")
    parser.add_argument("--eval_task", default=None)
    parser.add_argument("--save_pred_audio", default=None, help="whether to save predict audio or not")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    user_args = vars(args)
    logger.info(f"Processing task: \nglobal args: {user_args}")
    t = Pipeline.create(**user_args)
    t.run()

if __name__ == "__main__":
    main()
