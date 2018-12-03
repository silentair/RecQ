'''embedding'''
import numpy as np

def SVD_embeding(ui_matrix,keep_rate=0.9,factor_num=None):
    u,s,v = np.linalg.svd(ui_matrix)

    if factor_num is None:
        threshhold = np.sum(s**2) * keep_rate
        sigma_sum_k = 0
        k = 0
        for i in s:
            sigma_sum_k = sigma_sum_k + i**2
            k = k + 1
            if sigma_sum_k >= threshhold:
                break
    else:
        k = factor_num

    user_vec = np.mat(u[:,:k]).tolist()
    item_vec = np.mat(v[:,:k]).tolist()
    
    return user_vec,item_vec