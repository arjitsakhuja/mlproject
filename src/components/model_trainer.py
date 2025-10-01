import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": xgb.XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Support Vector Regressor": SVR(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )
            logging.info("Training and Evaluating the Models")

            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best model found: {best_model_name} with score {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Best model saved")

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
