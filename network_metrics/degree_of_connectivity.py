# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:44:57 2021

@author: john
"""
def degree_of_connectivity(Adj_matrix):
    degrees = Adj_matrix[:,:].sum(axis = 1)
    normalized_degrees = degrees/degrees.sum()
    return normalized_degrees