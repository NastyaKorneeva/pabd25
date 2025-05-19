import argparse
from flask import Flask, render_template, request
from logging.config import dictConfig
import joblib


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


# Сохранение модели
MODEL_NAME = "models/linear_regression_model.pkl"


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


def predict_with_model(app, total_meters):
    price = app.config["model"].predict(
        [
            [
                total_meters,
            ]
        ]
    )[0]
    price = int(price)
    return {"status": "success", "data": price}


def predict_with_heuristic(app, total_meters):
    return {"status": "success", "data": total_meters * app.config["price_for_meter"]}


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
def process_numbers():
    data = request.get_json()

    app.logger.info(f"Request data: {data}")
    try:
        total_meters = float(data["area"])
    except ValueError:
        return {"status": "error", "data": "Ошибка парсинга данных"}

    if app.config["use_heuristic"]:
        return predict_with_heuristic(app, total_meters)
    elif app.config["model"] is not None:
        return predict_with_model(app, total_meters)
    print(app.config["model"])
    return {"status": "error", "data": "Ошибка создания прогноза"}


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    parser.add_argument("--heurisic", help="Use heuristic", default="false")
    args = parser.parse_args()

    try:
        app.config["model"] = joblib.load(args.model)
    except FileNotFoundError:
        app.config["model"] = None

    app.config["price_for_meter"] = 300_000
    app.config["use_heuristic"] = args.heurisic.lower() == "true"
    app.logger.info(f"Use model: {args.model}")
    app.run(debug=True)
