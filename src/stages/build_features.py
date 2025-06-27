import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import setup_logging, load_params


def main():
    params = load_params()
    DATA_RAW_PATH = Path("data/raw")
    DATA_PROCESSED_PATH = Path("data/processed")
    DATA_PROCESSED_PATH.mkdir(exist_ok=True, parents=True)

    df = pd.concat([pd.read_csv(f) for f in DATA_RAW_PATH.glob("*.csv")])

    # Обработка данных
    df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
    df = df[["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]]

    # Фильтрация
    df = df[
        (df["price"] < params["data"]["price_limit"])
        & (df["total_meters"] < params["data"]["area_limit"])
    ]

    # Feature engineering
    df["rooms_1"] = df["rooms_count"] == 1
    df["rooms_2"] = df["rooms_count"] == 2
    df["rooms_3"] = df["rooms_count"] == 3
    df["first_floor"] = df["floor"] == 1
    df["last_floor"] = df["floor"] == df["floors_count"]

    # Разделение данных
    train_df, test_df = train_test_split(
        df.drop(columns=["floor", "rooms_count"]).set_index("url_id"),
        test_size=params["train"]["test_size"],
        shuffle=False,
    )

    train_df.to_csv(DATA_PROCESSED_PATH / "train.csv")
    test_df.to_csv(DATA_PROCESSED_PATH / "test.csv")


if __name__ == "__main__":
    setup_logging()
    main()
