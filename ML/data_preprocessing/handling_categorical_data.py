import pandas as pd

# Generate CSV data
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'class_label']
print("original data:\n", df)

# Mapping ordinal features to numerical values (ascending order)
# ( python dictionary: {'key':value} )
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
print("df_size_numerical\n", df)

# Mapping numerical values back to the ordinal features
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print("df_size_ordinal\n", df)