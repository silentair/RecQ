import numpy as np
import tensorflow as tf

'''parameters
ui_mat = ui_matrix
avg = rating_mean
n_i = i_num
n_u = u_num
n_k = k
lr = 0.01
lamb_u = 0.02
lamb_v = 0.02
'''

#ra = SVD_plusplus_pred(ui_matrix,rating_mean,i_num,u_num,k)
#print(ra[2][13:30])

#data = ac.get_FilmTrust()

#ui_matrix,is_rating = ac.get_User_Item_matrix(data)

#ra = mf.MF_pred(ui_matrix,is_rating)
#print(ra[2][13:30])

#ilist = [13,14,15,16,1,18,19,20]
#ra_2_to_ilist = svd.SVD_pred_with_sim(ui_matrix,2,ilist)
#print(ra_2_to_ilist)

#import Algorithm.embedding as eb

#u,v = eb.SVD_ItemUser2vec(ui_matrix)

from Configurations.config import Config
from Tool.Dao import RationgDao
from Algorithm.rating.BasicMF import BasicMatrixFactorization
from Tool.File import FileIO
from Algorithm.rating.SVD import SingularValueDecomposition
from Algorithm.rating.SVDplusplus import SingularValueDecompositionPlusPlus
from Algorithm.rating.UserKNN import UserKnn
from Algorithm.rating.ItemKNN import ItemKnn

config = Config('./configurations/ItemKnn.conf')
dao = RationgDao(config)


#mf = BasicMatrixFactorization(config,dao)
#mf.showRecommenderInfo()
#mf.Training()
#mf.Testing()

#svd = SingularValueDecomposition(config,dao)
#svd.Training()
#svd.Testing()

#svd_pp = SingularValueDecompositionPlusPlus(config,dao)
#svd_pp.Training()
#svd_pp.Testing()

#uknn = UserKnn(config,dao)
#uknn.Training()
#uknn.Testing()

iknn = ItemKnn(config,dao)
#iknn.Training()
iknn.Testing()