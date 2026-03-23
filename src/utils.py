import os
import sys
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