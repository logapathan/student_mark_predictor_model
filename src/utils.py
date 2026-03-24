import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill
import numpy as np
import src.exceptions as CustomException
from src.logger import logging

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models: dict,params) -> dict:
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            params_for_model = params[list(models.keys())[i]]

            gs= GridSearchCV(model,params_for_model,cv=5)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)                  
            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {e}")
        raise CustomException(e, sys)