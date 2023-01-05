# %%
import pickle
import itertools
import numpy as np
from numpy import ma
import pandas as pd
import sklearn.metrics
from math import log
from tqdm import tqdm

from importlib import reload
import utils;reload(utils)
from utils import pmap_multi
# Human
# Edge：graph_nx_human_L_surface_10k.pickle
# X: HCPS403_xtract_bp_10k_L.pickle
# Y: HCPS403_myelin_NoOutlier_10k_L.pickle

# Macaque
# Edge: graph_nx_macaque_L_surface_10k.pickle
# X: LN8_xtract_bp_10k_L.pickle
# Y: Macaque_actual_myelin_L.pickle

def pickle_load(fn):
    with open(fn, "rb") as input_file:
        return np.squeeze(np.array(pickle.load(input_file)))

fn_list = ['graph_nx_human_L_surface_10k.pickle', 'HCPS403_xtract_bp_10k_L.pickle', 'HCPS403_myelin_NoOutlier_10k_L.pickle', 'graph_nx_macaque_L_surface_10k.pickle', 'LN8_xtract_bp_10k_L.pickle', 'Macaque_actual_myelin_L.pickle']
fn_list = ['/hpc-cache-pfs/home/yufeng/data/hm/'+fn for fn in fn_list]
ah, xh, yh, am, xm, ym = map(pickle_load, fn_list)

ym[~np.isfinite(ym)] = 0  # ym has nan

# print(ah.shape, xh.shape, yh.shape, am.shape, xm.shape, ym.shape)
# (27969, 2) (403, 9368, 42) (403, 9368) (26934, 2) (8, 9027, 42) (9027,)

# import scipy.io
# mat = scipy.io.loadmat('data/KL_matrix_L.mat')
# arr = mat['KL_matrix_L'].transpose()
# sim = np.power(arr, -4)

# pd.DataFrame(yh.transpose()).describe()


# %%
def standardization(data):
    row_sum = np.sum(data, axis=1, keepdims=True)
    res = np.divide(data, row_sum, out=np.zeros_like(data), where=row_sum!=0)
    return res
     
def log2(A):
    # mask 0 for log(0)
    return ma.log2(A).filled(0)

# KL
def my_kl(A, B):
    # A (m,p); B (n,p); out (m,n)
    A = standardization(A)
    B = standardization(B)
    return A * log2(A) @ np.ones_like(B.transpose()) - A @ log2(B.transpose()) +\
        np.ones_like(A) @ (B * log2(B)).transpose() - log2(A) @ B.transpose()

def my_p(x1, y1, x2, y2, r=-1.2):
    # use 2 predict 1
    arr = my_kl(x1, x2)  # 9027, 9368
    sim = np.power(arr, r)
    sim[~np.isfinite(sim)] = 0
    p_hm = pd.DataFrame([sim @ y2 / np.sum(sim, axis=1) , y1]).T.corr().to_numpy()[0,1]
    return p_hm

res = {}


# %%
# m2m

def pair_p(i, j):
    return i,j,my_p(xm[i], ym, xm[j], ym, -4)
    
start, end = 0, xm.shape[0]
res['m2m'] = pmap_multi(pair_p, itertools.permutations(range(start, end), 2), backend='multiprocessing')



# %%
# h2m
def pair_p(i, j):
    return i,j,my_p(xm[i], ym, xh[j], yh[j], -4)

a, b = range(xm.shape[0]), range(xh.shape[0])
res['h2m'] = pmap_multi(pair_p, itertools.product(a, b), backend='multiprocessing')



# %%
# m2h
def pair_p(i, j):
    return i,j,my_p(xh[i], yh[i], xm[j], ym, -4)

a, b = range(xh.shape[0]), range(xm.shape[0])
res['m2h'] = pmap_multi(pair_p, itertools.product(a, b), backend='multiprocessing')



# %%
# h2h
def pair_p(i, j):
    return i,j,my_p(xh[i], yh[i], xh[j], yh[j], -4)

start, end = 0, xh.shape[0] # xh.shape[0] 10
res['h2h'] = pmap_multi(pair_p, itertools.permutations(range(start, end), 2), backend='multiprocessing')


# %%
with open(r"res.pickle", "wb") as output_file:
    pickle.dump(res, output_file)

# %%
# # 任意2h相关性
# # num_h = yh.shape[0]
# # res = np.zeros([num_h,num_h])
# s, e = 50, 100
# for i,j in tqdm(itertools.permutations(range(s, e), 2)):
#     p = my_p(xh[i], yh[i], xh[j], yh[j], -4)
#     res[i][j] = round(p, 3)
# res

# %%
# # 复现0.53

# xh_mean = xh.mean(axis=0) # 9368, 42
# yh_mean = yh.mean(axis=0) # 9368
# xm_mean = xm.mean(axis=0) # 9027, 42
# my_p(xm_mean, ym, xh_mean, yh_mean, -4)


