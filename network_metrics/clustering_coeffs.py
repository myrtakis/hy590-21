# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:20:02 2021

@author: john
"""
from itertools import combinations
import numpy as np
import pandas as pd

## Construct adjency matrix
def form_adj_matrix():
    pass

## i neuranl id A Adj Matrix 
def comp_clustering_coef(A,i):
    k_i = A[i,:].sum() 
    local_clustering_coef = sum([ A[i,j]* sum([A[j,k]*A[k,i] for k in range(A.shape[0])]) for j in range(A.shape[0])])
    #or
    #local_clusterign_coef = sum([ A[i,j] * (A[j,:]*A[:,i]).sum() for j in range(90)])
    local_clustering_coef *= 1/(k_i*(k_i -1))
    return local_clustering_coef

## Compute Local Clustering Coefficients
def local_clustering_coeffs(Adj_matrix):
    Adj_matrix = np.ones((90,90))
    #clustering_coefs = []
    for i in range(Adj_matrix.shape[0]):
        yield comp_clustering_coef(Adj_matrix,i)
        #clustering_coefs.append(comp_clustering_coef(A,i))
    #return clustering_coefs
    
