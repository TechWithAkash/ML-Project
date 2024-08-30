# all the common functionality come here

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
# import pickle
import dill  
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            # pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException('Error saving object', e)
