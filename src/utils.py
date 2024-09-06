# all the common functionality come here

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from src.exception import CustomException
# # import pickle
# import dill  
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV
# def save_object(file_path,obj):
#     try:
#         dir_path = os.path.dirname(file_path)

#         os.makedirs(dir_path,exist_ok=True)

#         with open(file_path, 'wb') as file_obj:
#             dill.dump(obj, file_obj)
#             # pickle.dump(obj, file_obj)

#     except Exception as e:
#         raise CustomException('Error saving object', e)

# def evaluate_models(X_train,y_train,X_test,y_test,models,param):

    # try:
    #     reports = {}
    #     for i in range(len(list(models))):
    #         model = list(models.values())[i]
    #         para = param[list(models.keys())[i]]
    #         # model.fit(X_train, y_train) # taining model
 
    #         gs = GridSearchCV(model,para,cv=3)
    #         gs.fit(X_train, y_train)

    #         model.set_params(**gs.best_params_)
    #         model.fit(X_train, y_train)

    #         y_train_pred = model.predict(X_train)
    #         y_test_pred = model.predict(X_test)
    #         train_model_score= r2_score(y_train,y_train_pred)
    #         test_model_score= r2_score(y_test,y_test_pred)
    #         reports[list(models.keys())[i]] = test_model_score
    #     return reports

    # except Exception as e:
    #     raise CustomException('Error evaluating models', e)


import pickle
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f'Model saved to {file_path}')
    except Exception as e:
        raise CustomException(f"Error saving the model: {str(e)}")

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            if model_name in param:
                # Perform GridSearchCV for models with parameters
                gs = GridSearchCV(estimator=model, param_grid=param[model_name], scoring=make_scorer(r2_score), cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_estimator = gs.best_estimator_
                model = best_estimator
            else:
                model.fit(X_train, y_train)

            # Predict and score
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_report[model_name] = score
            logging.info(f'{model_name} R2 Score: {score}')
        
        return model_report

    except Exception as e:
        raise CustomException(f"Error evaluating models: {str(e)}")
