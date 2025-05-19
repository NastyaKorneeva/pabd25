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
import os

import cianparser
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TEST_SIZE = 0.2
N_ROOMS = 1  # just for the parsing step
MODEL_NAME = "decision_tree_reg_1.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """
    moscow_parser = cianparser.CianParser(location="Москва")

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"data/raw/{n_rooms}_{t}.csv"
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

    train_df.to_csv("data/processed/train.csv")
    test_df.to_csv("data/processed/test.csv")


def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y = train_df["price"]
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.values, y)

    logging.info(f"Train {model} and save to {model_path}")

    joblib.dump(model, model_path)


def test_model(model_path):
    """Test model with new data"""
    test_df = pd.read_csv("data/processed/test.csv")
    train_df = pd.read_csv("data/processed/train.csv")
    X_test = test_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_test = test_df["price"]
    X_train = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_train = train_df["price"]
    model = joblib.load(model_path)
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    logging.info(f"Test model. MSE: {mse:.2f}")
    logging.info(f"Test model. RMSE: {rmse:.2f}")
    logging.info(f"Test model. MAE: {mae:.2f}")
    logging.info(f"Test model. R2 train: {r2_train:.2f}")
    logging.info(f"Test model. R2 test: {r2_test:.2f}")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
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
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    parser.add_argument(
        "-p", "--parse_data", help="Flag to parse new data", action="store_true"
    )
    args = parser.parse_args()

    test_size = float(args.split)
    assert 0.0 <= test_size <= 0.5
    model_path = os.path.join("models", args.model)

    if args.parse_data:
        parse_cian(args.n_rooms)
    preprocess_data(test_size)
    train_model(model_path)
    test_model(model_path)
