import pandas as pd
import numpy as np
from numpy.linalg import norm
#from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
# Reading files
'''
# users
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
# rating
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
# items
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 
		  'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 
		  'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
		  'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('../data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
'''
def cosine_similarity(v1, v2):
	v1 = np.array(v1)
	v2 = np.array(v2)
	return np.dot(v1, v2) / (norm(v1) * norm(v2))



r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols)
ratings_base = pd.read_csv('../data/ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('../data/ml-100k/ua.test', sep='\t', names=r_cols)

n_users = df.user_id.unique().shape[0]
n_items = df.movie_id.unique().shape[0]

train_data_matrix = np.zeros((n_users, n_items))
for row in ratings_base.itertuples():
	train_data_matrix[row[1]-1, row[2]-1] = row[3]

test_data_matrix = np.zeros((n_users, n_items))
for row in ratings_test.itertuples():
	test_data_matrix[row[1]-1, row[2]-1] = row[3]


user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

