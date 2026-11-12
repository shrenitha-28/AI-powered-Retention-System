import sys
import pandas as pd
import numpy as np

from logcode import setup_logging
logger=setup_logging("null_values")


def handle_null_values(df):
    try:
        d=pd.DataFrame({'TotalCharges':df['TotalCharges']})
        logger.info(f"Total Charges: {d}")
        logger.info(f"Shape of data:{d.shape}")

        #Apply Methods
        #Mean
        d['Mean']=d['TotalCharges'].fillna(df['TotalCharges'].mean())

        #Median
        d['Median']=d['TotalCharges'].fillna(df['TotalCharges'].median())

        #Mode
        d['Mode']=d['TotalCharges'].fillna(df['TotalCharges'].mode().iloc[0])

        #Forward Fill
        d['ForwardFill']=d['TotalCharges'].ffill()

        #Backward Fill
        d['BackwardFill']=d['TotalCharges'].bfill()

        #Linear Interpolation
        d['LinearInterpolate'] = d['TotalCharges'].interpolate(method='linear')

        #Quadratic Interpolate
        d['QuadraticInterpolate'] = d['TotalCharges'].interpolate(method='quadratic')

        #Arbitrary
        d['TotalCharges_arbitrary'] = d['TotalCharges'].fillna(555)

        #EOD
        d['TotalCharges_eod'] = d['TotalCharges'].fillna(d['TotalCharges'].mean() + 3 * d['TotalCharges'].std())

        #Constant
        d['TotalCharges_Const'] = np.where(d['TotalCharges'].isnull(), 0, 1)

        #Print d
        logger.info(f"The dataset is:{d}")

        logger.info(f"Sd of below columns")
        logger.info(f"Original:{d['TotalCharges'].std()}")
        logger.info(f"Mean:{d['Mean'].std()}")
        logger.info(f"Median:{d['Median'].std()}")
        logger.info(f"Mode:{d['Mode'].std()}")
        logger.info(f"Forward Fill:{d['ForwardFill'].std()}")
        logger.info(f"Backward Fill:{d['BackwardFill'].std()}")
        logger.info(f"Arbitrary:{d['TotalCharges_arbitrary'].std()}")
        logger.info(f"EOD:{d['TotalCharges_eod'].std()}")
        logger.info(f"Constant:{d['TotalCharges_Const'].std()}")

        d['RandomSample'] = d['TotalCharges'].copy()
        s = d['TotalCharges'].dropna().sample(d['TotalCharges'].isnull().sum(), random_state=42)
        s.index = d[d['TotalCharges'].isnull()].index
        d.loc[d['TotalCharges'].isnull(), 'RandomSample'] = s

        logger.info(f"Random Sample:{d['RandomSample'].std()}")

        df['TotalCharges']=d['Mode']
        logger.info(f"Original:{df['TotalCharges'].std()}")

        return df


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error occured due to {er_ty} and Error msg:{er_msg} in line {er_line.tb_lineno}")