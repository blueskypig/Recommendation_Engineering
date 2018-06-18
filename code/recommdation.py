import pandas as pd

# Reading files
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
print(users.head())