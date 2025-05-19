import logging
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseTrainer:
    def __init__(self, model_path, train_df, test_df, random_state=42, cv=5):
        """Initialize the linear regression trainer"""
        self.random_state = random_state
        self.best_model = None
        self.cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.train_df = train_df
        self.test_df = test_df

    def _get_X_y(self, df):
        """Prepare features and target"""
        X = df[
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
        y = df["price"]
        return X, y

    def _evaluate(self, X, y, dataset_name):
        """Evaluate model performance"""
        y_pred = self.best_model.predict(X)

        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }

        logging.info(f"{dataset_name} metrics:")
        logging.info(f"MSE: {metrics['mse']:.2f}")
        logging.info(f"RMSE: {metrics['rmse']:.2f}")
        logging.info(f"MAE: {metrics['mae']:.2f}")
        logging.info(f"R2: {metrics['r2']:.2f}")

        return metrics


class LinRegTrainer(BaseTrainer):
    def __init__(self, model_path, train_df, test_df, random_state=42, cv=5):
        """Initialize the linear regression trainer"""
        super().__init__(model_path, train_df, test_df, random_state, cv)

        self.scaler = StandardScaler()

    def _get_model_pipeline(self, estimator):
        """Create pipeline with scaler and estimator"""
        return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    def _perform_grid_search(self, X, y):
        """Perform grid search to find best model and hyperparameters"""
        # Define models and their parameter grids
        models = {
            "ridge": {
                "estimator": Ridge(),
                "params": {
                    "estimator__alpha": [0.01, 0.1, 1, 10, 100],
                    "estimator__fit_intercept": [True, False],
                },
            },
            "lasso": {
                "estimator": Lasso(),
                "params": {
                    "estimator__alpha": [0.01, 0.1, 1, 10, 100],
                    "estimator__fit_intercept": [True, False],
                },
            },
            "elasticnet": {
                "estimator": ElasticNet(),
                "params": {
                    "estimator__alpha": [0.01, 0.1, 1, 10],
                    "estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "estimator__fit_intercept": [True, False],
                },
            },
        }

        best_score = -np.inf
        best_model = None

        for model_name, model_config in models.items():
            pipeline = self._get_model_pipeline(model_config["estimator"])
            grid = GridSearchCV(
                pipeline,
                model_config["params"],
                cv=self.cv,
                scoring="r2",
                n_jobs=-1,
                verbose=1,
            )

            grid.fit(X, y)

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                logging.info(f"New best model: {model_name} with R2: {best_score:.4f}")

        return best_model

    def train(self):
        """Train the best linear regression model"""
        X_train, y_train = self._get_X_y(self.train_df)

        # Perform grid search to find best model
        self.best_model = self._perform_grid_search(X_train, y_train)

        # Final training on full dataset
        self.best_model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(self.best_model, self.model_path)
        logging.info(f"Trained model saved to {self.model_path}")

        # Return training metrics
        return self._evaluate(X_train, y_train, "Train")

    def test(self):
        """Test the trained model"""
        X_test, y_test = self._get_X_y(self.test_df)

        # Load the model if not already loaded
        if self.best_model is None:
            self.best_model = joblib.load(self.model_path)

        return self._evaluate(X_test, y_test, "Test")


class CatBoostTrainer(BaseTrainer):
    def __init__(self, model_path, train_df, test_df, random_state=42, cv=5):
        """Initialize the CatBoost trainer"""
        super().__init__(model_path, train_df, test_df, random_state, cv)

        self.cat_features = self._identify_categorical_features()

    def _identify_categorical_features(self):
        """Identify categorical features in the dataset"""
        # Assuming binary features (0/1) are categorical for CatBoost
        categoricals = []
        for col in self.train_df.columns:
            if col in ["rooms_1", "rooms_2", "rooms_3", "first_floor", "last_floor"]:
                categoricals.append(col)
        return categoricals

    def _perform_grid_search(self, X, y):
        """Perform grid search to find best CatBoost parameters"""
        model = CatBoostRegressor(
            random_seed=self.random_state, verbose=False, cat_features=self.cat_features
        )

        param_grid = {
            "iterations": [500, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128],
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X, y)

        logging.info(f"Best CatBoost parameters: {grid.best_params_}")
        logging.info(f"Best R2 score: {grid.best_score_:.4f}")

        return grid.best_estimator_

    def train(self):
        """Train the CatBoost model"""
        X_train, y_train = self._get_X_y(self.train_df)

        # Perform grid search to find best model
        self.best_model = self._perform_grid_search(X_train, y_train)

        # Final training on full dataset (with early stopping)
        eval_set = (
            X_train.iloc[:100],
            y_train.iloc[:100],
        )  # Small subset for early stopping
        self.best_model.fit(
            X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=100
        )

        # Save the trained model
        self.best_model.save_model(self.model_path)
        logging.info(f"Trained model saved to {self.model_path}")

        # Return training metrics
        return self._evaluate(X_train, y_train, "Train")

    def test(self):
        """Test the trained model"""
        X_test, y_test = self._get_X_y(self.test_df)

        # Load the model if not already loaded
        if self.best_model is None:
            self.best_model = CatBoostRegressor()
            self.best_model.load_model(self.model_path)

        return self._evaluate(X_test, y_test, "Test")
