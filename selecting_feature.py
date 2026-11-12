import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import sys


from sklearn.feature_selection import VarianceThreshold

from logcode import setup_logging
logger=setup_logging("selecting_feature")

def feature(X_train,X_test,y_train,y_test):
    try:
        #Initially the data
        logger.info(f"Initially X_train:{X_train.shape}, \n X_test:{X_test.shape}")
        logger.info(f"The columns are: {X_train.columns}")

        #Constant Technique
        const_reg=VarianceThreshold(threshold=0.0)
        const_reg.fit(X_train)
        logger.info(f"The bad columns are: {X_train.columns[~const_reg.get_support()]}")

        #Quasi Constant Technique
        quasi_reg=VarianceThreshold(threshold=0.1)
        quasi_reg.fit(X_train)
        logger.info(f"The bad columns are: {X_train.columns[~quasi_reg.get_support()]}")

        # # Pearson Correlation
        # values=[]
        # for i in X_train.columns:
        #     values.append(pearsonr(X_train[i],y_train))
        # values_array=np.array(values)
        # p_value=pd.Series(values_array[:,1],index=X_train.columns)
        # plt.figure(figsize=(5, 3))
        # p_value.plot.bar()
        # plt.show()
        for i in X_train.columns:
            corr, p_value = pearsonr(X_train[i], y_train)
            logger.info(f"Feature: {i}, Correlation: {corr}, P-value: {p_value}")

        #Mutual Information
        mi_scores = mutual_info_classif(X_train,y_train,random_state=42)
        mi_scores = pd.Series(mi_scores,index=X_train.columns)
        for feature, score in mi_scores.items():
            logger.info(f"Feature: {feature}, Mutual Information Score: {score}")

        #Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100,random_state=42)
        rf.fit(X_train, y_train)
        importance = pd.Series(rf.feature_importances_,index=X_train.columns)
        for feature, score in importance.items():
            logger.info(f"Feature: {feature}, Importance: {score}")

        return X_train,X_test,y_train,y_test


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")