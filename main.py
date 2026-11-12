import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

from logcode import setup_logging
from pyparsing import exceptions

logger=setup_logging("main")

import warnings
warnings.filterwarnings("ignore")

from null_values import handle_null_values
from variable_transformation import transformation
from selecting_feature import feature
from catencoding import encode
from selecting_cat_features import feature_cat
from allmodels import training

class Customer:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(path)

            #Details of dataset like: Size, First 5 rows
            logger.info(f"The size of the dataset is:{self.df.shape}")
            logger.info(f"The first 5 rows are:\n{self.df.head(5)}")
            logger.info(f"The values of SIM for each category:\n{self.df['SIM'].value_counts()}")

            #Convert Churn into 0 and 1
            self.df['Churn']=self.df['Churn'].map({'Yes':1,'No':0})

            #Convert Senior Citizen from numerical to categorical
            self.df['SeniorCitizen']=self.df['SeniorCitizen'].map({0:'No',1:'Yes'})
            logger.info(f"The first 5 rows are:\n{self.df.head(5)}")
            logger.info(f"The data types of the columns are:")
            for i in self.df.columns:
                logger.info(i)
                logger.info(self.df[i].dtype)

            #Drop CustomerID
            self.df=self.df.drop(['customerID'],axis=1)
            logger.info(f"The size of the dataset is:{self.df.shape}")
            logger.info(f"The first 5 rows are:\n{self.df.head(5)}")

            # Divide data into X and y
            self.X = self.df.drop(['Churn'], axis=1)
            self.y = self.df['Churn']

            # Checking data in X and y
            logger.info(f"The no.of columns in X is:{self.X.shape}")
            logger.info(f"The no.of columns in y is:{self.y.shape}")

            # Divide data into train test and split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=45)
            logger.info(f"The shape of X_train is:{self.X_train.shape}")
            logger.info(f"The shape of X_test is:{self.X_test.shape}")
            logger.info(f"The shape of y_train is:{self.y_train.shape}")
            logger.info(f"The shape of y_test is:{self.y_test.shape}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def handle_missing_values(self):
        try:
            logger.info(f"The data types of the columns are:")
            for i in self.X_train.columns:
                logger.info(i)
                logger.info(f"Train:{self.X_train[i].dtype}")
                logger.info(f"Test:{self.X_test[i].dtype}")

            #Change TotalCharges from Str to float64
            self.X_train['TotalCharges']=pd.to_numeric(self.X_train['TotalCharges'], errors='coerce')
            self.X_test['TotalCharges'] = pd.to_numeric(self.X_test['TotalCharges'], errors='coerce')
            logger.info(f"The data type of TotalCharges is:{self.X_train['TotalCharges'].dtype}")
            logger.info(f"The data type of TotalCharges is:{self.X_test['TotalCharges'].dtype}")

            #Before Handling
            logger.info("=====Before Handling=========")
            logger.info("=========Train data details===========")
            logger.info(self.X_train.shape)
            logger.info(self.X_train.isnull().sum())

            logger.info("=========Test data details===========")
            logger.info(self.X_test.shape)
            logger.info(self.X_test.isnull().sum())

            self.X_train=handle_null_values(self.X_train)
            self.X_test=handle_null_values(self.X_test)

            #After Handling
            logger.info("=====After Handling=========")
            logger.info("=========Train data details===========")
            logger.info(self.X_train.shape)
            logger.info(self.X_train.isnull().sum())

            logger.info("=========Test data details===========")
            logger.info(self.X_test.shape)
            logger.info(self.X_test.isnull().sum())

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def data_separation(self):
        try:
            #Dividing into Numerical and Categorical
            logger.info(f"The data types of the columns are:")
            for i in self.X_train.columns:
                logger.info(i)
                logger.info(f"Train:{self.X_train[i].dtype}")
                logger.info(f"Test:{self.X_test[i].dtype}")

            #Numerical Columns
            self.X_train_num = self.X_train.select_dtypes(exclude='str')
            self.X_test_num = self.X_test.select_dtypes(exclude='str')

            #Categorical Columns
            self.X_train_cat = self.X_train.select_dtypes(include='str')
            self.X_test_cat = self.X_test.select_dtypes(include='str')

            #Checking about data
            logger.info("=========Train data details===========")
            logger.info(f"Train Numerical columns : {len(self.X_train_num.columns)} : \n : {self.X_train_num.columns}")
            logger.info(f"Train Categorical columns : {len(self.X_train_cat.columns)} : \n : {self.X_train_cat.columns}")

            logger.info("=========Test data details===========")
            logger.info(f"Test Numerical columns : {len(self.X_test_num.columns)} : \n : {self.X_test_num.columns}")
            logger.info(f"Test Categorical columns : {len(self.X_test_cat.columns)} : \n : {self.X_test_cat.columns}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def variable_outlier(self):
        try:
            logger.info("Before VT Outlier...")
            logger.info(self.X_train_num.shape)
            logger.info(self.X_test_num.shape)
            self.X_train_num,self.X_test_num=transformation(self.X_train_num, self.X_test_num)
            logger.info(f"After VT Outlier...")
            logger.info(self.X_train_num.shape)
            logger.info(self.X_test_num.shape)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")
    def feature_selection(self):
        try:
            logger.info("Before Feature Selection...")
            logger.info(self.X_train_num.shape)
            logger.info(self.X_test_num.shape)
            self.X_train_num, self.X_test_num,self.y_train,self.y_test=feature(self.X_train_num, self.X_test_num,self.y_train,self.y_test)
            logger.info("After Feature Selection...")
            logger.info(self.X_train_num.shape)
            logger.info(self.X_test_num.shape)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def encoding(self):
        try:
            logger.info("Before Encoding...")
            logger.info(self.X_train_cat.shape)
            logger.info(f"The data types of X_train_cat is:{self.X_train_cat.dtypes}")
            logger.info(self.X_test_cat.shape)
            logger.info(f"The data types of X_test_cat is:{self.X_test_cat.dtypes}")
            self.X_train_cat,self.X_test_cat=encode(self.X_train_cat,self.X_test_cat)
            logger.info("After Encoding...")
            logger.info(self.X_train_cat.shape)
            logger.info(f"The data types of X_train_cat is:{self.X_train_cat.dtypes}")
            logger.info(self.X_test_cat.shape)
            logger.info(f"The data types of X_test_cat is:{self.X_test_cat.dtypes}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def feature_sel_cat(self):
        try:
            logger.info("Before Feature Selection...")
            logger.info(self.X_train_cat.shape)
            logger.info(self.X_test_cat.shape)
            self.X_train_cat, self.X_test_cat, self.y_train, self.y_test = feature_cat(self.X_train_cat, self.X_test_cat,
                                                                                   self.y_train, self.y_test)
            logger.info("After Feature Selection...")
            logger.info(self.X_train_cat.shape)
            logger.info(self.X_test_cat.shape)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

    def data_combining(self):
        try:

            # Combine numerical and categorical features
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            # Merging both Numerical and Categorical columns
            self.X_train = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.X_test = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info("After combining numerical and categorical features")
            logger.info(f"X_train Shape : {self.X_train.shape}")
            logger.info(f"X_test Shape : {self.X_test.shape}")
            logger.info(f"X_train Columns:\n{self.X_train.columns}")
            logger.info(f"X_test Columns:\n{self.X_test.columns}")
            logger.info(self.X_train.isnull().sum())
            logger.info(self.X_test.isnull().sum())

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")
    def data_balancing(self):
        try:
            logger.info(f"No.of Good Columns:{sum(self.y_train==1)}")
            logger.info(f"No.of Bad Columns:{sum(self.y_train==0)}")
            logger.info(f"{self.y_train.shape}")

            # Before ROS
            logger.info("Before ROS")
            logger.info(self.X_train.shape)
            logger.info("====Before ROS training======")
            for i in self.X_train.columns:
                logger.info(i)
                logger.info(self.X_train[i].value_counts())
            logger.info(self.y_train.value_counts())
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=42)

            self.X_train_ros, self.y_train_ros = ros.fit_resample(
                self.X_train,
                self.y_train
            )
            # After ROS
            logger.info("After ROS")
            logger.info(self.X_train_ros.shape)
            logger.info(self.y_train_ros.value_counts())

            logger.info(f"No.of Good Columns:{sum(self.y_train_ros == 1)}")
            logger.info(f"No.of Bad Columns:{sum(self.y_train_ros == 0)}")
            logger.info(f"{self.y_train.shape}")
            logger.info("=====Before training data=====")
            for i in self.X_train_ros.columns:
                logger.info(i)
                logger.info(self.X_train_ros[i].value_counts())

            training(self.X_train_ros,self.y_train_ros,self.X_test,self.y_test)

            logger.info("Training Dataset")
            logger.info(self.X_train_ros.shape)
            logger.info(self.X_train_ros.head(5))

            # Training Dataset
            train_dataset = pd.concat(
                [
                    self.X_train_ros.reset_index(drop=True),
                    self.y_train_ros.reset_index(drop=True)
                ],
                axis=1
            )

            # Testing Dataset
            test_dataset = pd.concat(
                [
                    self.X_test.reset_index(drop=True),
                    self.y_test.reset_index(drop=True)
                ],
                axis=1
            )

            train_dataset.to_csv("train_dataset.csv", index=False)
            test_dataset.to_csv("test_dataset.csv", index=False)

            logger.info("Training and Testing datasets saved successfully.")
            logger.info(f"Training Dataset Shape: {train_dataset.shape}")
            logger.info(f"Testing Dataset Shape: {test_dataset.shape}")


        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")



r=Customer("churn.csv")
r.handle_missing_values()
r.data_separation()
r.variable_outlier()
r.feature_selection()
r.encoding()
r.feature_sel_cat()
r.data_combining()
r.data_balancing()