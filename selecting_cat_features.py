import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from logcode import setup_logging
logger=setup_logging("selecting_cat_features")

def feature_cat(X_train, X_test, y_train, y_test):
    try:
        # Chi-Square
        chi_score, p_value = chi2(X_train, y_train)
        for feature, score, p in zip(X_train.columns, chi_score, p_value):
            logger.info(f"Feature: {feature}, Chi2 Score: {score}, P-value: {p}")

        #Feature: gender P - value: 0.3440164895442098
        #Feature: PhoneService P-value: 0.7280745221269217
        #Feature: MultipleLines_No phone service P-value: 0.2904714666799304
        #Feature: SIM_BSNL P-value: 0.8910258422751081
        #Feature: SIM_Jio P-value: 0.9284391749340801
        #Feature: SIM_Vodafone P-value: 0.784982516996902

        #So remove gender, PhoneService, MultipleLines_No phone service, SIM_BSNL, SIM_Jio, SIM_Vodafone

        # Mutual Information
        mi_scores = mutual_info_classif(X_train,y_train,random_state=42)
        mi_scores = pd.Series(mi_scores, index=X_train.columns)
        for feature, score in mi_scores.items():
            logger.info(f"Feature: {feature}, Mutual Information Score: {score}")

        #Feature: gender, Mutual Information Score: 0.0
        #Feature: PhoneService, Mutual Information Score: 0.006279266606747269
        #Feature: MultipleLines_No phone service, Mutual Information Score: 0.0
        #Feature: OnlineBackup_Yes, Mutual Information Score: 0.0
        #Feature: PaymentMethod_Credit card (automatic), Mutual Information Score: 0.0
        #Feature: SIM_Jio, Mutual Information Score: 0.0

        # Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100,random_state=42)
        rf.fit(X_train, y_train)
        importance = pd.Series(rf.feature_importances_,index=X_train.columns)
        for feature, score in importance.items():
            logger.info(f"Feature: {feature}, Importance: {score}")

        drop_columns = [
            "gender",
            "PhoneService",
            "MultipleLines_No phone service",
        ]

        X_train = X_train.drop(columns=drop_columns)
        X_test = X_test.drop(columns=drop_columns)

        return X_train,X_test,y_train,y_test

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")

