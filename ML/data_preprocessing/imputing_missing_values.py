import pandas as pd
from io import StringIO

# Generate CSV data
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print("df:\n", df)

# Check the number of null in each column
print(df.isnull().sum())

# Eliminating samples or features with missing values
df_dropNaN_row = df.dropna(axis=0)
print("df_dropNaN_row:\n", df_dropNaN_row)

df_dropNaN_column = df.dropna(axis=1)
print("df_dropNaN_column:\n", df_dropNaN_column)

# Only drop rows where all columns are NaN
df_all = df.dropna(how='all')
print("df_all:\n", df_all)

# Drop rows that have not at least 4 non-NaN values
df_thresh = df.dropna(thresh=4)
print("df_thresh:\n", df_thresh)

# Only drop rows where NaN appear in specific columns (here: 'C')
df_subset = df.dropna(subset=['C'])
print("df_subset:\n", df_subset)


### Imputing missing values
from sklearn.impute import SimpleImputer
import numpy as np

# Check NaN were replaced by?
imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputed_data = imr.fit_transform(df.values)
print("imputed_data:\n", imputed_data)