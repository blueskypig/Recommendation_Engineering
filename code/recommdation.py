import pandas as pd
import numpy as np
from numpy.linalg import norm
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
ratings_base = pd.read_csv('../data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')


