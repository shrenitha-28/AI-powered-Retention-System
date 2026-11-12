import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from logcode import setup_logging
logger = setup_logging("all_models")


import pickle
from sklearn.model_selection import GridSearchCV
from scipy.stats.distributions import nbinom
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve

def knn(X_train,y_train,X_test,y_test):
  global knn_reg
  knn_reg = KNeighborsClassifier(n_neighbors=5)
  knn_reg.fit(X_train,y_train)
  global knn_predictions
  knn_predictions = knn_reg.predict(X_test)
  logger.info(f"KNN Test Accuracy \n : {accuracy_score(y_test , knn_predictions)}")
  logger.info(f"KNN Confusion Matrix \n : {confusion_matrix(y_test,knn_predictions)}")
  logger.info(f"KNN Classification report \n : {classification_report(y_test,knn_predictions)}")

def nb(X_train,y_train,X_test,y_test):
  global nb_reg
  nb_reg = GaussianNB()
  nb_reg.fit(X_train,y_train)
  global nb_predictions
  nb_predictions = nb_reg.predict(X_test)
  logger.info(f"Naive Bayes Test Accuracy \n : {accuracy_score(y_test , nb_predictions)}")
  logger.info(f"Naive Bayes Confusion Matrix \n : {confusion_matrix(y_test,nb_predictions)}")
  logger.info(f"Naive Bayes Classification report \n : {classification_report(y_test,nb_predictions)}")


def lr(X_train,y_train,X_test,y_test):
  global lr_reg
  lr_reg = LogisticRegression()
  lr_reg.fit(X_train,y_train)
  global lr_predictions
  lr_predictions = lr_reg.predict(X_test)
  logger.info(f"Logistic Regression Test Accuracy \n : {accuracy_score(y_test , lr_predictions)}")
  logger.info(f"Logistic Regression Confusion Matrix \n : {confusion_matrix(y_test,lr_predictions)}")
  logger.info(f"Logistic Regression Classification report \n : {classification_report(y_test,lr_predictions)}")


def dt(X_train,y_train,X_test,y_test):
  global dt_reg
  dt_reg = DecisionTreeClassifier(criterion="entropy")
  dt_reg.fit(X_train,y_train)
  global dt_predictions
  dt_predictions = dt_reg.predict(X_test)
  logger.info(f"Decision Tree Test Accuracy \n : {accuracy_score(y_test , dt_predictions)}")
  logger.info(f"Decision Tree Confusion Matrix \n : {confusion_matrix(y_test,dt_predictions)}")
  logger.info(f"Decision Tree Classification report \n : {classification_report(y_test,dt_predictions)}")

def rf(X_train,y_train,X_test,y_test):
  global rf_reg
  rf_reg = RandomForestClassifier(criterion="entropy",n_estimators=5)
  rf_reg.fit(X_train,y_train)
  global rf_predictions
  rf_predictions = rf_reg.predict(X_test)
  logger.info(f"Random Forest Test Accuracy \n : {accuracy_score(y_test , rf_predictions)}")
  logger.info(f"Random Forest Confusion Matrix \n : {confusion_matrix(y_test,rf_predictions)}")
  logger.info(f"Random Forest Classification report \n : {classification_report(y_test,rf_predictions)}")

def ada(X_train,y_train,X_test,y_test):
  t = LogisticRegression()
  global ada_reg
  ada_reg = AdaBoostClassifier(estimator=t , n_estimators=5)
  ada_reg.fit(X_train,y_train)
  global ada_predictions
  ada_predictions = ada_reg.predict(X_test)
  logger.info(f"Adaboost Test Accuracy \n : {accuracy_score(y_test , ada_predictions)}")
  logger.info(f"Adaboost Confusion Matrix \n : {confusion_matrix(y_test,ada_predictions)}")
  logger.info(f"Adaboost Classification report \n : {classification_report(y_test,ada_predictions)}")

def GB(X_train,y_train,X_test,y_test):
  global gb_reg
  gb_reg = GradientBoostingClassifier(n_estimators=5)
  gb_reg.fit(X_train,y_train)
  global gb_predictions
  gb_predictions = gb_reg.predict(X_test)
  logger.info(f"Gradient Boosting Test Accuracy \n : {accuracy_score(y_test , gb_predictions)}")
  logger.info(f"Gradient Boosting Confusion Matrix \n : {confusion_matrix(y_test,gb_predictions)}")
  logger.info(f"Gradient Boosting Classification report \n : {classification_report(y_test,gb_predictions)}")

def xgb_(X_train,y_train,X_test,y_test):
  global xgb_reg
  xgb_reg = XGBClassifier(n_estimator = 5)
  xgb_reg.fit(X_train,y_train)
  global xgb_predictions
  xgb_predictions = xgb_reg.predict(X_test)
  logger.info(f"Xtreme Gradient Boosting Test Accuracy \n : {accuracy_score(y_test , xgb_predictions)}")
  logger.info(f"Xtreme Gradient Boosting Confusion Matrix \n : {confusion_matrix(y_test,xgb_predictions)}")
  logger.info(f"Xtreme Gradient Boosting Classification report \n : {classification_report(y_test,xgb_predictions)}")

def common(X_train,y_train,X_test,y_test):
  logger.info("============KNN==========")
  knn(X_train,y_train,X_test,y_test)
  logger.info("============Naive Bayes==========")
  nb(X_train,y_train,X_test,y_test)
  logger.info("============Logistic Regression==========")
  lr(X_train,y_train,X_test,y_test)
  logger.info("============Decision Tree==========")
  dt(X_train,y_train,X_test,y_test)
  logger.info("============Random Forest==========")
  rf(X_train,y_train,X_test,y_test)
  logger.info("============Adaboost==========")
  ada(X_train,y_train,X_test,y_test)
  logger.info("============Gradient Boosting==========")
  GB(X_train,y_train,X_test,y_test)
  logger.info("============Xtreme Gradient Boosting==========")
  xgb_(X_train,y_train,X_test,y_test)

def training(X_train, y_train, X_test, y_test):
  try:

    # common(X_train,y_train,X_test,y_test)
    #
    # knn_fpr, knn_tpr, knn_threshold = roc_curve(y_test, knn_predictions)
    # nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, nb_predictions)
    # lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, lr_predictions)
    # dt_fpr, dt_tpr, dt_threshold = roc_curve(y_test, dt_predictions)
    # rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, rf_predictions)
    # ada_fpr, ada_tpr, ada_threshold = roc_curve(y_test, ada_predictions)
    # gb_fpr, gb_tpr, gb_threshold = roc_curve(y_test, gb_predictions)
    # xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(y_test, xgb_predictions)
    #
    # plt.figure(figsize=(5, 3))
    # plt.title("Best Model selection")
    #
    # plt.plot(knn_fpr, knn_tpr, label='knn', color='r')
    # plt.plot(nb_fpr, nb_tpr, label='NB', color='g')
    # plt.plot(lr_fpr, lr_tpr, label='LR', color='black')
    # plt.plot(dt_fpr, dt_tpr, label='DT', color='blue')
    # plt.plot(rf_fpr, rf_tpr, label='RF', color='yellow')
    # plt.plot(ada_fpr, ada_tpr, label='ADA')
    # plt.plot(gb_fpr, gb_tpr, label='GB')
    # plt.plot(xgb_fpr, xgb_tpr, label='XGB')
    #
    # plt.legend()
    # plt.show()

    #Finding Best parameters
    # lr = LogisticRegression(random_state=42)
    # parameters_list = {'penalty': ['l1', 'l2', 'elasticnet', None],
    #                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                    'solver': ['liblinear', 'saga'],
    #                    'max_iter': [1000]
    #                    }
    # grid = GridSearchCV(
    #   estimator=lr,
    #   param_grid=parameters_list,
    #   cv=5,
    #   scoring="accuracy",
    #   n_jobs=-1
    # )
    # grid.fit(X_train, y_train)
    # # Best Parameters
    # logger.info("Best Parameters:")
    # logger.info(grid.best_params_)
    # logger.info("\nBest Cross Validation Accuracy:")
    # logger.info(grid.best_score_)
    # # Best Model
    # best_lr = grid.best_estimator_
    # # Prediction
    # y_pred = best_lr.predict(X_test)
    # # Evaluation
    # logger.info("\nTest Accuracy:")
    # logger.info(accuracy_score(y_test, y_pred))
    # logger.info("\nConfusion Matrix:")
    # logger.info(confusion_matrix(y_test, y_pred))
    # logger.info("\nClassification Report:")
    # logger.info(classification_report(y_test, y_pred))


    # Final Logistic Regression Model
    # Final Logistic Regression Model
    final_lr = LogisticRegression(
      C=0.1,
      penalty="l2",
      solver="saga",
      max_iter=1000,
      random_state=42
    )

    # Train the model on ROS-balanced training data
    final_lr.fit(X_train, y_train)

    # Predict on test data
    y_pred = final_lr.predict(X_test)

    # Predict probabilities for ROC-AUC
    y_prob = final_lr.predict_proba(X_test)[:, 1]

    # Evaluation
    logger.info("Final Logistic Regression Model")

    logger.info(f"Accuracy:{accuracy_score(y_test, y_pred)}")

    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    logger.info(f"=======Saving the Model===================")
    with open("Customer.pkl", "wb") as t:
      pickle.dump(final_lr, t)

  except Exception as e:
    er_ty, er_msg, er_lineno = sys.exc_info()
    logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")