import pandas as pd

# Generate CSV data
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'class_label']
print("original data:\n", df)


# ### Encoding class labels via numpy
# import numpy as np

# # generate dictionary of labels
# class_mapping = {label: idx for idx, label in enumerate(np.unique(df['class_label']))}
# print(f"class mapping: {type(class_mapping)}\n", class_mapping)

# # Mapping class labels to integers
# df['class_label'] = df['class_label'].map(class_mapping)
# print("df_encoding_labels\n", df)

# # Mapping integers back to class labels
# inv_class_mapping = {v: k for k, v in class_mapping.items()}
# df['class_label'] = df['class_label'].map(inv_class_mapping)
# print("df_encoding_labels\n", df)


## Encoding class labels via scikit-learn
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['class_label'].values)
print(y)

y = class_le.inverse_transform(y)
print(y)
