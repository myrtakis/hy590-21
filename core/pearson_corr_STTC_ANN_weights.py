# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:16:52 2021

@author: john
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns; sns.set_theme()


fold_name = "fold_10"
import sys
sys.exit()

#os.chdir('C:\\Users\\john\\Desktop\\hy590-21\\environment\\configs\\fully_connected\\1622537625.328412\\'+fold_name)
new_model = tf.keras.models.load_model('C:\\Users\\john\\Desktop\\hy590-21\\environment\\configs\\fully_connected\\1622048185.470065\\'+fold_name+'\\model_epoch_100.hdf5')

weights = new_model.weights[0]
   
sttc_weights = pd.read_csv("C:\\Users\\john\\Desktop\\hy590-21\\data\\l4_l23_sttc_matrix.csv")

#map_l23_id_ind = {id_ : i for i,id_ in enumerate(sttc_weights.loc[:,"l23"].unique())}

#map_l4_id_ind = {id_ : i for i,id_ in enumerate(sttc_weights.loc[:,"l4"].unique())}

#sttc_weights.loc[(sttc_weights["l4"] == 132) & (sttc_weights["l23"] == 1680),:].STTC

#sttc_correlation_matrix = np.zeros((len(map_l4_id_ind),len(map_l23_id_ind)))

#sttc_correlation_matrix = [[sttc_weights.loc[(sttc_weights["l4"] == l4_id) & (sttc_weights["l23"] == l23_id),:].STTC for l23_id in map_l23_id_ind] for l4_id in map_l4_id_ind]

sttc_correlation_matrix = sttc_weights.pivot_table(values="STTC", index="l4", columns="l23")


ax = sns.heatmap(sttc_correlation_matrix, cmap="YlGnBu")
plt.figure()
ax = sns.heatmap(weights, cmap="YlGnBu")
plt.figure()
plt.show()
plt.figure()
#r_coefs = np.corrcoef(sttc_correlation_matrix, weights, rowvar = False)

r_coefs_pearson = [stats.pearsonr(sttc_correlation_matrix.iloc[:,i], weights[:,i])[0] for i in range(weights.shape[1])]

r_coefs_spearman = [stats.spearmanr(sttc_correlation_matrix.iloc[:,i], weights[:,i])[0] for i in range(weights.shape[1])]

pd.DataFrame(r_coefs_pearson).to_csv("C:\\Users\\john\\Desktop\\hy590-21\\pearson_coefs_STTC_DNN_weights.csv")

pd.DataFrame(r_coefs_spearman).to_csv("C:\\Users\\john\\Desktop\\hy590-21\\spearman_coefs_STTC_DNN_weights.csv")

# the histogram of the data
n, bins, patches = plt.hist(r_coefs_pearson, 50, density=True, facecolor='g', alpha=0.75, label = 'pearson coef')

#n, bins, patches = plt.hist(r_coefs_spearman, 50, density=True, facecolor='g', alpha=0.75, label = 'spearman coef')

plt.xlabel('pearson coefs', size = 12)
plt.ylabel('freq.', size = 12)
plt.title('associations btw STTC - ANN weights - ref neuron l23', size = 16)
plt.text(-0.08, 20, r'$\mu='+str(np.mean(r_coefs_pearson))+',\ \sigma='+str(np.std(r_coefs_pearson))+'$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()