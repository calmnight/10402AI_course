import numpy as np
import pandas as pd

### load Wine dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
print(df_wine.tail())

### partitioning dataset
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

print("Features:\n", X)
print("class:\n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ### Min-max normalization
# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# X_train_mms = mms.fit_transform(X_train)
# X_test_mms = mms.transform(X_test)

# print("X_train_mms:\n", X_train_mms)
# print("X_test_mms:\n", X_test_mms)


# ### Standardization
# from sklearn.preprocessing import StandardScaler
# stdsc = StandardScaler()
# X_train_stdsc = stdsc.fit_transform(X_train)
# X_test_stdsc = stdsc.transform(X_test)

# print("X_train_stdsc:\n", X_train_stdsc)
# print("X_test_stdsc:\n", X_test_stdsc)


### Robust scaling
from sklearn.preprocessing import RobustScaler
rs = RobustScaler(quantile_range=(25.0, 75.0))
X_train_rs = rs.fit_transform(X_train)
X_test_rs = rs.transform(X_test)

print("X_train_rs:\n", X_train_rs)
print("X_test_rs:\n", X_test_rs)