"""
This is full life cycle for ml model.
Linear regression with 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import datetime
import glob
import logging
import os
from logging.config import dictConfig
from pathlib import Path

import cianparser
import pandas as pd
from sklearn.model_selection import train_test_split

from trainers import CatBoostTrainer, LinRegTrainer
import yaml

TEST_SIZE = 0.2
N_ROOMS = 1  # just for the parsing step
MODEL_NAME = "decision_tree_reg_1.pkl"


dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "train.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

DATA_PATH = Path(__file__).parent.parent / "data"
DATA_RAW_PATH = DATA_PATH / "raw"
DATA_RAW_PATH.mkdir(exist_ok=True, parents=True)
DATA_PROCESSED_PATH = DATA_PATH / "processed"
DATA_PROCESSED_PATH.mkdir(exist_ok=True, parents=True)


def parse_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """
    moscow_parser = cianparser.CianParser(location="Москва")

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = DATA_RAW_PATH / f"{n_rooms}_{t}.csv"
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)

    df.to_csv(csv_path, encoding="utf-8", index=False)


def preprocess_data(test_size):
    """
    Filter, sort and remove duplicates
    """
    raw_data_path = "./data/raw"
    file_list = glob.glob(raw_data_path + "/*.csv")
    logging.info(f"Preprocess_data. Use files to train: {file_list}")
    df = pd.read_csv(file_list[0])
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        df_i = pd.DataFrame(data)
        df = pd.concat([df, df_i], axis=0)

    df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
    df = (
        df[["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]]
        .set_index("url_id")
        .sort_index()
    )

    df.drop_duplicates(inplace=True)
    df = df[df["price"] < 100_000_000]
    df = df[df["total_meters"] < 100]

    df["rooms_1"] = df["rooms_count"] == 1
    df["rooms_2"] = df["rooms_count"] == 2
    df["rooms_3"] = df["rooms_count"] == 3
    df["first_floor"] = df["floor"] == 1
    df["last_floor"] = df["floor"] == df["floors_count"]
    df.drop(columns=["floor", "rooms_count"], inplace=True)

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    logging.info(f"Preprocess_data. train_df: {len(train_df)} samples")
    train_head = "\n" + str(train_df.head())
    logging.info(train_head)
    logging.info(f"Preprocess_data. test_df: {len(test_df)} samples")
    test_head = "\n" + str(test_df.head())
    logging.info(test_head)

    train_df.to_csv(DATA_PROCESSED_PATH / "train.csv")
    test_df.to_csv(DATA_PROCESSED_PATH / "test.csv")


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def infer_model_type(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    model_folder = model_path.parent.stem
    match model_folder:
        case "linear_regression":
            return LinRegTrainer
        case "catboost":
            return CatBoostTrainer
        case _:
            raise ValueError("unknown model type")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    params = load_params()["train"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test size, from 0 to 0.5",
        default=TEST_SIZE,
    )
    parser.add_argument(
        "-n", "--n_rooms", help="Number of rooms to parse", type=int, default=N_ROOMS
    )
    parser.add_argument("-m", "--model", help="Model name", default=params["model"])
    parser.add_argument(
        "-p", "--parse_data", help="Flag to parse new data", action="store_true"
    )
    args = parser.parse_args()

    test_size = float(args.split)
    assert 0.0 <= test_size <= 0.5
    model_path = os.path.join("models", args.model)

    any_files = next(DATA_RAW_PATH.iterdir(), False)
    if args.parse_data or not any_files:
        print("go")
        parse_cian(args.n_rooms)
    preprocess_data(test_size)

    train_df = pd.read_csv(DATA_PROCESSED_PATH / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED_PATH / "test.csv")

    trainer_cls = infer_model_type(model_path)
    trainer = trainer_cls(model_path=model_path, train_df=train_df, test_df=test_df)

    train_metrics = trainer.train()
    test_metrics = trainer.test()
