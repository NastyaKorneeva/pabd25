import argparse
from flask import Flask, render_template, request
from logging.config import dictConfig

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
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

import joblib

# Сохранение модели
MODEL_NAME = "models/linear_regression_v3.pkl"


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
def process_numbers():

    data = request.get_json()

    app.logger.info(f"Requst data: {data}")
    try:
        total_meters = float(data["area"])
        floors_count = int(data["total_floors"])
        rooms_1 = int(data["rooms"]) == 1
        rooms_2 = int(data["rooms"]) == 2
        rooms_3 = int(data["rooms"]) == 3
        first_floor = int(data["floor"]) == 1
        last_floor = int(data["floor"]) == floors_count
    except ValueError:
        return {"status": "error", "data": "Ошибка парсинга данных"}

    price = app.config["model"].predict(
        [
            [
                total_meters,
                floors_count,
                rooms_1,
                rooms_2,
                rooms_3,
                first_floor,
                last_floor,
            ]
        ]
    )[0]
    price = int(price)
    return {"status": "success", "data": price}


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()

    app.config["model"] = joblib.load(args.model)
    app.logger.info(f"Use model: {args.model}")
    app.run(debug=True)
