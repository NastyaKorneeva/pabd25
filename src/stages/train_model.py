import pandas as pd
from pathlib import Path
from trainers import CatBoostTrainer, LinRegTrainer
from utils import setup_logging, load_params


def infer_model_type(model_path):
    model_folder = Path(model_path).parent.stem
    return LinRegTrainer if model_folder == "linear_regression" else CatBoostTrainer


def main():
    params = load_params()
    DATA_PROCESSED_PATH = Path("data/processed")

    train_df = pd.read_csv(DATA_PROCESSED_PATH / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED_PATH / "test.csv")

    trainer_cls = infer_model_type(params["train"]["model"])
    trainer = trainer_cls(
        model_path=params["train"]["model"], train_df=train_df, test_df=test_df
    )

    train_metrics = trainer.train()
    test_metrics = trainer.test()


if __name__ == "__main__":
    setup_logging()
    main()
