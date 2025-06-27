from logging.config import dictConfig
import yaml


def setup_logging():
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


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
