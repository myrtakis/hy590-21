from itertools import combinations
import numpy as np
import pandas as pd
from stat_tests.tests import z_score
import os
import ray

# A validating example: for the pair (1, 1105) the STTC = -0.0032, CtrlGrpMean = 0.00038 CtrlGrpStDev = 0.023 and NullSTTC 0.026


def sttc(measurements_df, dt, n_jobs=1, write_dir=None):
    ray.init()

    assert measurements_df.shape[1] >= 2
    neuron_pairs = list(combinations(list(measurements_df.columns), 2))
    neuron_pairs = [neuron_pairs[0] for _ in range(100)]

    col_names = ['NeuronA', 'NeuronB', 'STTC', 'CtrlGrpMean', 'CtrlGrpStDev', 'NullSTTC', 'zscore']

    results = []

    for partition_pairs in np.split(np.array(neuron_pairs), n_jobs):
        results.append(sttc_for_pairs.remote(measurements_df, dt, partition_pairs))

    final_res = []
    for r in results:
        final_res.extend(ray.get(r))
    sttc_matrix = pd.DataFrame(final_res, columns=col_names)

    if write_dir is not None:
        if not os.path.isdir('results'):
            os.mkdir('results')
        sttc_matrix.to_csv(os.path.join(write_dir, 'sttc_matrix.csv'), index=False)

    return sttc_matrix


@ray.remote
def sttc_for_pairs(measurements_df, dt, pairs):
    sttc_mat = []
    for i, p in enumerate(pairs):
        neuronA = measurements_df.loc[:, p[0]]
        neuronB = measurements_df.loc[:, p[1]]
        print('Calculating STTC for pair', i, '/', len(pairs))
        observed_val = compute_corr(neuronA.values, neuronB.values, dt)
        CtrlGrpMean, CtrlGrpStDev, NullSTTC = compute_null_sttc(neuronA.values, neuronB.values, dt)
        zscore = z_score(observed_val, CtrlGrpMean, CtrlGrpStDev)
        sttc_mat.append([neuronA.name, neuronB.name, observed_val, CtrlGrpMean, CtrlGrpStDev, NullSTTC, zscore])
    return sttc_mat
    

def compute_corr(neuronA, neuronB, dt):
    # This function returns the sttc score
    assert neuronA.shape[0] == neuronB.shape[0]

    ### Try for dt =8 20 big sensitivity

    neuronA_shifted_plus = np.array(neuronA)
    neuronB_shifted_minus = np.array(neuronB)
    ### Find the assembled porpotion of timestamp within +- lag fraction over all spike events per neuron A B respectively
    for lag in range(dt):
        neuronA_shifted_plus += custom_shift(neuronA_shifted_plus, 1, fill_value=0)
        neuronB_shifted_minus += custom_shift(neuronB_shifted_minus, -1, fill_value=0)

    neuronA_shifted_plus[neuronA_shifted_plus != 0] = 1
    neuronB_shifted_minus[neuronB_shifted_minus != 0] = 1

    T_A_plus = neuronA_shifted_plus.sum()/neuronA.shape[0]
    T_B_minus = neuronB_shifted_minus.sum()/neuronB.shape[0]

    ### Number of splkes found within the lagged interval per direction
    P_A_Bminus = ((neuronA == 1) & (neuronB_shifted_minus == 1)).sum()/neuronA.sum()
    P_B_Aplus = ((neuronB == 1) & (neuronA_shifted_plus == 1)).sum()/neuronB.sum()

    section1 = (P_A_Bminus - T_B_minus) / (1 - P_A_Bminus*T_B_minus)
    section2 = (P_B_Aplus - T_A_plus) / (1 - P_B_Aplus*T_A_plus)
    STTC_A_B = 0.5 * (section1 + section2)
    return STTC_A_B


def compute_null_sttc(neuronA, neuronB, dt):        
    # This function performs 500 random circular shifts and calls compute corr for each one of them
    # it returns CtrlGrpMean, CtrlGrpStDev and a random NullSTTC value from the 500
    num_of_shifts = 500
    null_distribution = []
    np.random.seed(0)
    for boostratps in range(num_of_shifts):
        random_shift_A = np.random.randint(neuronA.shape[0])
        neuronA_randomly_shfited = np.roll(neuronA, random_shift_A)
        null_distribution.append(compute_corr(neuronA_randomly_shfited, neuronB, dt))
    CtrlGrpMean = np.mean(null_distribution)
    CtrlGrpStDev = np.std(null_distribution)
    return CtrlGrpMean, CtrlGrpStDev, null_distribution[np.random.randint(len(null_distribution))]


def custom_shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result








