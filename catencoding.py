import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import pickle
import sys

from logcode import setup_logging
logger=setup_logging("cat_encoding")

def encode(X_train,X_test):
    try:
        #Display data
        logger.info(X_train.columns)
        # logger.info(X_test.columns)
        for i in X_train.columns:
            logger.info(i)
            logger.info(X_train[i].nunique())
            logger.info(X_test[i].value_counts())
        #The below columns has 2 unique values: Gender, SeniorCitizen,Partner,Dependents,PhoneService,PaperlessBilling
        # label=pd.DataFrame({'Gender':X_train['Gender'],'SeniorCitizen':X_train['SeniorCitizen'],'Partner':X_train['Partner'],'Dependents':X_train['Dependents'],'PhoneService':X_train['PhoneService'],'PaperlessBilling':X_train['PaperlessBilling']})

        #Label Encoding
        X_train['gender']=X_train['gender'].map({'Male':0,'Female':1})
        X_train['SeniorCitizen']=X_train['SeniorCitizen'].map({'Yes':1,'No':0})
        X_train['Partner']=X_train['Partner'].map({'Yes':1,'No':0})
        X_train['Dependents']=X_train['Dependents'].map({'Yes':1,'No':0})
        X_train['PhoneService']=X_train['PhoneService'].map({'Yes':1,'No':0})
        X_train['PaperlessBilling']=X_train['PaperlessBilling'].map({'Yes':1,'No':0})

        X_test['gender']=X_test['gender'].map({'Male':0,'Female':1})
        X_test['SeniorCitizen']=X_test['SeniorCitizen'].map({'Yes':1,'No':0})
        X_test['Partner']=X_test['Partner'].map({'Yes':1,'No':0})
        X_test['Dependents']=X_test['Dependents'].map({'Yes':1,'No':0})
        X_test['PhoneService']=X_test['PhoneService'].map({'Yes':1,'No':0})
        X_test['PaperlessBilling']=X_test['PaperlessBilling'].map({'Yes':1,'No':0})


        for i in X_train.columns:
            logger.info(i)
            logger.info(X_train[i].dtypes)

        #One-hot encoding
        logger.info("Before One hot Encoding:")
        logger.info(f"=====Train Dataset======")
        logger.info(f"Training data shape is {X_train.shape} : \n Training columns are: \n{X_train.columns}")
        logger.info(f"=====Test Dataset======")
        logger.info(f"Testing data shape is {X_test.shape} : \n Testing columns are: \n{X_test.columns}")
        one_hot_columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaymentMethod', 'SIM']
        one_hot = OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)
        one_hot.fit(X_train[one_hot_columns])

        with open("onehot_encoder.pkl", "wb") as f:
            pickle.dump(one_hot, f)
        result1=one_hot.transform(X_train[one_hot_columns])
        result2=one_hot.transform(X_test[one_hot_columns])
        t1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out(one_hot_columns))
        t2 = pd.DataFrame(data=result2, columns=one_hot.get_feature_names_out(one_hot_columns))
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)

        X_train = pd.concat([X_train, t1], axis=1)
        X_test = pd.concat([X_test, t2], axis=1)

        X_train = X_train.drop(one_hot_columns, axis=1)
        X_test = X_test.drop(one_hot_columns, axis=1)

        logger.info("After One hot Encoding:")
        logger.info(f"=====Train Dataset======")
        logger.info(f"Training data shape is {X_train.shape} : \n Training columns are: \n{X_train.columns}")
        logger.info(f"=====Test Dataset======")
        logger.info(f"Testing data shape is {X_test.shape} : \n Testing columns are: \n{X_test.columns}")

        #Ordinal Encoding
        contract_order = [["Month-to-month",
                           "One year",
                           "Two year"]]
        ord = OrdinalEncoder(categories=contract_order)
        ord.fit(X_train[['Contract']])
        with open("ordinal_encoder.pkl", "wb") as f:
            pickle.dump(ord, f)
        result_3 = ord.transform(X_train[['Contract']])
        result_4 = ord.transform(X_test[['Contract']])

        t3 = pd.DataFrame(data=result_3, columns=ord.get_feature_names_out() + "_res")
        t4 = pd.DataFrame(data=result_4, columns=ord.get_feature_names_out() + "_res")

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        t3.reset_index(drop=True, inplace=True)
        t4.reset_index(drop=True, inplace=True)

        X_train = pd.concat([X_train, t3], axis=1)
        X_test = pd.concat([X_test, t4], axis=1)

        X_train = X_train.drop(['Contract'], axis=1)
        X_test = X_test.drop(['Contract'], axis=1)

        X_train.reset_index(drop=True, inplace=True)
        X_train.reset_index(drop=True, inplace=True)

        logger.info("After Ordinal Encoding:")
        logger.info(f"=====Train Dataset======")
        logger.info(f"Training data shape is {X_train.shape} : \n Training columns are: \n{X_train.columns}")
        logger.info(f"=====Test Dataset======")
        logger.info(f"Testing data shape is {X_test.shape} : \n Testing columns are: \n{X_test.columns}")

        logger.info("Final Training Dataset")
        logger.info(f"Training data shape  is {X_train.shape} : \n Training columns are: \n{X_train.columns}\n")
        logger.info(X_train.head(10))
        return X_train,X_test


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")