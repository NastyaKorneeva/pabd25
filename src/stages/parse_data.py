import datetime
from pathlib import Path
import cianparser
import pandas as pd
from utils import setup_logging, load_params
from logging.config import dictConfig
import yaml
import pandas as pd
import numpy as np


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


def generate_fake():
    np.random.seed(42)

    data = {
        "url": [
            f"https://www.cian.ru/sale/flat/{i}" for i in range(1, 101)
        ],  # 100 уникальных url_id
        "total_meters": np.round(
            np.random.uniform(30, 150, 100), 1
        ),  # Площадь от 30 до 150 м²
        "floor": np.random.randint(1, 25, 100),  # Этаж от 1 до 24
        "floors_count": np.random.randint(5, 25, 100),  # Всего этажей от 5 до 24
        "rooms_count": np.random.choice(
            [1, 2, 3, 4, 5], 100
        ),  # Кол-во комнат от 1 до 5
        "price": np.random.randint(2_000_000, 20_000_000, 100),  # Цена от 2 до 20 млн
    }
    data["floors_count"] = np.maximum(data["floors_count"], data["floor"])

    df = pd.DataFrame(data)
    df = df[["url", "total_meters", "floor", "floors_count", "rooms_count", "price"]]

    return df


def main():
    params = load_params()
    DATA_RAW_PATH = Path("data/raw")
    DATA_RAW_PATH.mkdir(exist_ok=True, parents=True)

    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = DATA_RAW_PATH / f"{params['data']['n_rooms']}_{t}.csv"
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(1,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary",
        },
    )
    if data:
        pd.DataFrame(data).to_csv(csv_path, index=False)
    else:
        generate_fake().to_csv(csv_path, index=False)


if __name__ == "__main__":
    setup_logging()
    main()
