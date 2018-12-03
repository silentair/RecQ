import numpy as np
import math

''' normalization '''
def Normalize(vector,param1,param2,method='min-max'):
    if method == 'min-max':
        maxVal = param1
        minVal = param2
        normalized_data = (vector - minVal) / (maxVal - minVal)
    elif method == 'z-score':
        mean = param1
        var = param2
        normalized_data = (vector - mean) / var
    else:
        print('please check your choose the correct normalization method')
        raise ValueError
    return normalized_data

def Denormalize(vector,param1,param2,method='min-max'):
    if method == 'min-max':
        maxVal = param1
        minVal = param2
        denormalized_data = vector * (maxVal - minVal) + minVal
    elif method == 'z-score':
        mean = param1
        var = param2
        denormalized_data = vector * var + mean
    else:
        print('please check your choose the correct normalization method')
        raise ValueError
    return denormalized_data

''' similarity '''
def Euclidean_sim(vector_a,vector_b):
    if len(vector_a) != len(vector_b):
        print('inputs must have same size!')
        raise ValueError
    return 1 / (1 + np.linalg.norm(vector_a-vector_b))

def Pearson_sim(vector_a,vector_b):
    if len(vector_a) != len(vector_b):
        print('inputs must have same size!')
        raise ValueError
    if np.std(vector_a)*np.std(vector_b) == 0:
        corr = 0
    else:
        corr = np.corrcoef(vector_a,vector_b)[0][1]
    return 0.5 * corr + 0.5

def Cosin_sim(vector_a,vector_b):
    if len(vector_a) != len(vector_b):
        print('inputs must have same size!')
        raise ValueError
    return np.sum(vector_a * vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

def Similarity(vector_a,vector_b,method):
    if method == 'Cosin':
        sim = Cosin_sim(vector_a,vector_b)
    elif method == 'Euclidean':
        sim = Euclidean_sim(vector_a,vector_b)
    elif method == 'Pearson':
        sim = Pearson_sim(vector_a,vector_b)
    else:
        print('unavailable similarity method!')
        raise ValueError
    return sim

''' evaluation '''
def MAE(pred,real):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return np.sum(np.abs(pred-real)) / len(pred)

def RMAE(pred,real):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return np.sqrt(np.sum((pred-real)**2) / len(pred))