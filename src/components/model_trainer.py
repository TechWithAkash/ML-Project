# import sys, os
# from dataclasses import dataclass
# from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
# from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor)
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score  # Import the correct r2_score function
# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info('Splitting Training and test input data...')
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1],
#             )

#             models = {
#                 'AdaBoostRegressor': AdaBoostRegressor(),
#                 'GradientBoostingRegressor': GradientBoostingRegressor(),
#                 'RandomForestRegressor': RandomForestRegressor(),
#                 'LinearRegression': LinearRegression(),
#                 'Ridge': Ridge(),
#                 'Lasso': Lasso(),
#                 'ElasticNet': ElasticNet(),
#                 'SGDRegressor': SGDRegressor(),
#                 'KNeighborsRegressor': KNeighborsRegressor(),
#                 'DecisionTreeRegressor': DecisionTreeRegressor(),
#                 'XGBRegressor': XGBRegressor(),
#             }

#             # performing Hyperparameter Tunning
#             logging.info('Performing Hyperparameter Tuning...')
#             # Hyperparameter tuning for each model
#             params = {
#                "Decision Tree":{
#                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
#                },
#                 "Random Forest":{
#                    'n_estimators':[100,200,300,400,500],
#                    'criterion':['gini','entropy'],
#                    'max_depth':[5,10,15,20,25,30],
#                    'min_samples_split':[2,5,10,15,20],
#                 },
#                 "XGBRegressor":{
#                    'n_estimators':[100,200,300,400,500],
#                    'learning_rate':[0.01,0.05,0.1,0.2,0.3],
#                    'max_depth':[3,5,7],
#                    'min_child_weight':[1,3,5],},

#                     "AdaBoostRegressor":{
#                    'n_estimators':[100,200,300,400,500],
#                    'learning_rate':[0.01,0.05,0.1,0.2,0.3],
#                    },
#                     "GradientBoostingRegressor":{
#                         'n_estimators':[100,200,300,400,500],
#                         'learning_rate':[0.01,0.05,0.1,0.2,0.3],
#                         'max_depth':[3,5,7],
#                         'min_child_weight':[1,3,5],
#                     }
#            }
        
#         except Exception as e:
        
#                  raise CustomException('Error initiating model trainer', e)

#         model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

#             # Get best model score from dict
#         best_model_score = max(model_report.values())

#             # Get the best model's name from dict
#         best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

#         best_model = models[best_model_name]

#         if best_model_score < 0.6:
#                     raise CustomException("No Best Model found..")

#                 # Save the best model
#         save_object(
#                     file_path=self.model_trainer_config.trained_model_file_path,
#                     obj=best_model
#                 )

#         predicted = best_model.predict(X_test)

#         score = r2_score(y_test, predicted)  # Use the r2_score function from sklearn.metrics
#         return score
        

            
import os
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
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
            logging.info('Splitting Training and test input data...')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'SGDRegressor': SGDRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
            }

            logging.info('Performing Hyperparameter Tuning...')
            params = {
                "DecisionTreeRegressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "RandomForestRegressor": {
                    'n_estimators': [100, 400, 500],
                    'criterion': ['squared_error', 'absolute_error'],
                },
                "XGBRegressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.2, 0.3],
                },
                "AdaBoostRegressor": {
                    'n_estimators': [100, 500],
                    'learning_rate': [0.01, 0.05, 0.3],
                },
                "GradientBoostingRegressor": {
                    'n_estimators': [300, 400, 500],
                    'learning_rate': [0.01, 0.05, 0.3],
                }
            }

        # Evaluate models with hyperparameter tuning
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Get the best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            print(f"Best Model score is {best_model_score} with model {best_model_name}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with sufficient score.")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict with the best model
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException('Error initiating model trainer', str(e))