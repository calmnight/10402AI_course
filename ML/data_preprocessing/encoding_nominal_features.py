import pandas as pd

# Generate CSV data
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'class_label']
print("original data:\n", df)


# ### Encoding nominal features (bad method)
# from sklearn.preprocessing import LabelEncoder
# X = df[['color', 'size', 'price']].values

# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# print("Results:\n", X)


### Encoding nominal features (good method)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = df[['color', 'size', 'price']].values

ct = ColumnTransformer(
    transformers=[('color_encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)
print("Results:\n", X_transformed)

feature_names = ct.get_feature_names_out()
print("Feature names:\n", feature_names)
