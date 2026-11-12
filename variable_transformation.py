import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer,QuantileTransformer
from scipy.stats import zscore
import sys
import pickle


from logcode import setup_logging
logger=setup_logging("variable_transformation")

def fun(df,var):
  plt.figure(figsize=(8,3))
  plt.title(var)

  plt.subplot(1,3,1)
  df[var].plot(kind='kde',label=var)

  plt.subplot(1,3,2)
  sns.boxplot(x=df[var])

  plt.subplot(1,3,3)
  stats.probplot(df[var],dist='norm',plot=plt)

  plt.show()

def transformation(X_train,X_test):
  try:
    # for i in X_train.columns:
    #   has_zero_or_negative = (X_train[i] <= 0).any()
    #   logger.info(f"The columns is {i} and has negative or zero values in it: {has_zero_or_negative}")
    #
    # for i in X_test.columns:
    #   has_zero_or_negative = (X_test[i] <= 0).any()
    #   logger.info(f"The columns is {i} and has negative or zero values in it: {has_zero_or_negative}")

    #Tenure has 0 or negative values in both train and test data
    #But MonthlyCharges and TotalCharges doesn't have 0 or negative values

    for i in X_train.columns:
      if not (X_train[i] <= 0).any():
        X_train[i+'_log']=np.log(X_train[i])

        X_train[i+'_reci']=1/X_train[i]

        X_train[i+'_sqrt']=np.sqrt(X_train[i])

        #Since TotalCharges is much larger than 1000
        if i!='TotalCharges':
          X_train[i+'_exp']=np.exp(X_train[i])

        pt1 = PowerTransformer(method='box-cox')
        X_train[i+'_boxcox']=pt1.fit_transform(X_train[[i]])
        X_test[i + '_boxcox'] = pt1.transform(X_test[[i]])

      pt2 = PowerTransformer(method='yeo-johnson')
      X_train[i+'_yeo_johnson'] = pt2.fit_transform(X_train[[i]])
      X_test[i + '_yeo_johnson'] = pt2.transform(X_test[[i]])

      qt = QuantileTransformer(
        output_distribution='normal',
        random_state=42
      )

      # Fit on training data
      X_train[i + '_quantile'] = qt.fit_transform(X_train[[i]])

      # Transform test data
      X_test[i + '_quantile'] = qt.transform(X_test[[i]])

      # Save QuantileTransformer for this column
      with open(f"{i}_quantile.pkl", "wb") as f:
        pickle.dump(qt, f)

      X_train[i+'_rank'] = X_train[i].rank()
    logger.info(f"The data shape of X_train is {X_train.shape}")
    logger.info(f"The data shape of X_test is {X_test.shape}")
    logger.info(f"The columns are:{X_train.columns}")
    logger.info(f"The columns are:{X_test.columns}")

    # for i in X_train.columns:
    #   fun(X_train,i)

    X_train=pd.DataFrame({'TotalCharges':X_train['TotalCharges_quantile'], 'Tenure':X_train['tenure_quantile'], 'MonthlyCharges':X_train['MonthlyCharges_quantile']})
    X_test=pd.DataFrame({'TotalCharges':X_test['TotalCharges_quantile'], 'Tenure':X_test['tenure_quantile'], 'MonthlyCharges':X_test['MonthlyCharges_quantile']})


    #Outliers
    X=pd.DataFrame({'TotalCharges':X_train['TotalCharges'], 'Tenure':X_train['Tenure'], 'MonthlyCharges':X_train['MonthlyCharges']})
    for i in X_train.columns:
      #Trimming
      q1=X_train[i].quantile(0.25)
      q3=X_train[i].quantile(0.75)

      iqr=q3-q1
      lower=q1-1.5*iqr
      upper=q3+1.5*iqr

      X[i + "_trim"] = np.where(X_train[i] > upper, upper,
                                          np.where(X_train[i] < lower, lower, X_train[i]))
      X_test[i]=np.where(X_test[i]>upper,upper,
                         np.where(X_test[i]<lower, lower, X_test[i]))

      #Mean+-std
      mean=X_train[i].mean()
      std=X_train[i].std()

      upper_limit=mean+3*std
      lower_limit=mean-3*std

      X[i+"_cap_with_mean"]=np.where(X_train[i] >= upper_limit, upper_limit,
                                           np.where(X_train[i]<lower_limit, lower_limit, X_train[i]))

      #5th and 95th percentile
      fifth_percentile=X_train[i].quantile(0.05)
      ninety_fifth_percentile=X_train[i].quantile(0.95)

      X[i+"_5th_95th"]=np.where(X_train[i]<fifth_percentile, fifth_percentile,
                                      np.where(X_train[i]>=ninety_fifth_percentile, ninety_fifth_percentile, X_train[i]))

      #z score
      z = np.abs(zscore(X_train[i]))
      upper_limit_value = X_train[i].mean() + 3 * X_train[i].std()
      lower_limit_value = X_train[i].mean() - 3 * X_train[i].std()
      X[i + "_zscore"] = np.where( z > 3, np.where(X_train[i] > upper_limit, upper_limit,lower_limit),X_train[i])

      # 10th and 90th percentile
      tenth_percentile = X_train[i].quantile(0.10)
      ninety_percentile = X_train[i].quantile(0.90)

      X[i + "_10th_90th"] = np.where(X_train[i] < tenth_percentile, tenth_percentile,
                                    np.where(X_train[i] >= ninety_percentile, ninety_percentile, X_train[i]))

    # for i in X.columns:
    #   fun(X,i)
    for i in X_train.columns:
      X_train[i]=X[i+'_trim']
      # fun(X_train,i)
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The shape of X_test is {X_test.shape}")
    return X_train,X_test
  except Exception as e:
    er_ty, er_msg, er_line = sys.exc_info()
    logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")
