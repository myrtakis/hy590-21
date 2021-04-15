import pandas as pd
import numpy as np


def clustering_coef(sttc_matrix, sig_threshold):
    adj_mat = adj_matrix(sttc_matrix, sig_threshold)
    c_coef_dict = {}
    for ind in adj_mat.index:
        neighbors_ids = adj_mat.columns[adj_mat.loc[ind] == 1]
        neighbors = adj_mat.loc[neighbors_ids, neighbors_ids]
        if len(neighbors) == 1:
            c_coef = 0
        else:
            c_coef = (neighbors.values.sum() / 2) / ((len(neighbors_ids) * (len(neighbors_ids)-1)) / 2)
        c_coef_dict[ind] = c_coef
    return pd.DataFrame(list(c_coef_dict.items()), columns=['NeuronID', 'CluCoef'])


def degree_of_connectivity(sttc_matrix, sig_threshold):
    adj_mat = adj_matrix(sttc_matrix, sig_threshold)
    doc = adj_mat.sum(axis=1) / adj_mat.shape[1]
    n_doc = pd.DataFrame(np.array([list(adj_mat.index), list(doc.values)], dtype=object).T, columns=['NeuronID', 'NormDoC'])
    return n_doc


# Helper Functions

def adj_matrix(sttc_matrix, sig_threshold):
    sig_pairs = sttc_matrix[sttc_matrix['zscore'] > 4]
    unique_pairs = {*sttc_matrix['NeuronA'].values, *sttc_matrix['NeuronB'].values}
    adj_mat = pd.DataFrame(0, index=unique_pairs, columns=unique_pairs)
    for p in sig_pairs.loc[:, ['NeuronA', 'NeuronB']].values:
        adj_mat.loc[p[0], p[1]] = 1
        adj_mat.loc[p[1], p[0]] = 1
    return adj_mat
