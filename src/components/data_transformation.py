import os
import sys
import pandas as pd
import numpy as np

from src.exceptions import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()

    def data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines created")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")


            preprocessor = ColumnTransformer([
                ('num_pipeline', numerical_pipeline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])



            # train_path,test_path = data_ingestion.DataIngestion().initiate_data_ingestion()
            # train_df = pd.read_csv(train_path)
            # test_df = pd.read_csv(test_path)
            # logging.info("Read train and test data as dataframe")

            return preprocessor
            

        except Exception as e:
            logging.error(f"Error occurred while initiating data transformation: {e}")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data as dataframe")

            preprocessor_obj = self.data_transformation()
            logging.info("Obtained preprocessor object")

            traget_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[traget_column_name], axis=1)
            target_feature_train_df = train_df[traget_column_name]

            input_feature_test_df = test_df.drop(columns=[traget_column_name], axis=1)
            target_feature_test_df = test_df[traget_column_name]
            logging.info("Split input and target features from train and test dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info("Applied preprocessing object on training dataframe")

            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Applied preprocessing object on test dataframe")

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Concatenated input and target features for train and test dataframe")

            save_object(
                file_path=self.data_transfromation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return train_arr, test_arr, self.data_transfromation_config.preprocessor_obj_file_path

            

        except Exception as e:
            logging.error(f"Error occurred while initiating data transformation: {e}")
            raise CustomException(e, sys)
        
