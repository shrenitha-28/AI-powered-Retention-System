import pandas as pd
import numpy as np

# 1. Load your CSV file
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Define the unique values you want to assign
values_to_assign = ['Jio', 'Airtel', 'BSNL', 'Vodafone']
num_categories = len(values_to_assign)
num_rows = len(df)

# 3. Calculate how many times each value must repeat to fill the dataset
# "num_rows // num_categories" handles perfect splits
# "num_rows % num_categories" handles any remaining rows evenly
base_repeats = num_rows // num_categories
remainder = num_rows % num_categories

# Create a balanced array of values
balanced_array = np.repeat(values_to_assign, base_repeats)
if remainder > 0:
    balanced_array = np.append(balanced_array, values_to_assign[:remainder])

# 4. Shuffle the array randomly so the equal distribution is mixed up
np.random.shuffle(balanced_array)

# 5. Assign the shuffled, equally distributed array to a new column
df['SIM'] = balanced_array

# 6. Save the updated DataFrame back to a CSV file
df.to_csv('churn.csv', index=False)


