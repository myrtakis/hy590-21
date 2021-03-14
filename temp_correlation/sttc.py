from itertools import combinations
import numpy as np
import pandas as pd

# A validating example: for the pair (1, 1105) the STTC = -0.0032, CtrlGrpMean = 0.00038 CtrlGrpStDev = 0.023 and NullSTTC 0.026


def sttc(measurements_df, dt):
    assert measurements_df.shape[1] >= 2
    neuron_pairs = combinations(list(measurements_df.columns), 2)
    sttc_matrix = []
    col_names = ['NeuronA', 'NeuronB', 'STTC', 'CtrlGrpMean', 'CtrlGrpStDev', 'NullSTTC']
    for c in neuron_pairs:
        neuron1 = measurements_df.loc[:, c[0]]
        neuron2 = measurements_df.loc[:, c[1]]
        observed_val = compute_corr(neuron1, neuron2, dt)
        CtrlGrpMean, CtrlGrpStDev, NullSTTC = compute_null_sttc(neuron1, neuron2, dt)
        sttc_matrix.append([neuron1.name, neuron2.name, observed_val, CtrlGrpMean, CtrlGrpStDev, NullSTTC])
    return pd.DataFrame(sttc_matrix, columns=col_names)


def compute_corr(neuron1, neuron2, dt):
    # This function returns the sttc score
    return 0


def compute_null_sttc(neuron1, neuron2, dt):
    CtrlGrpMean = None
    CtrlGrpStDev = None
    NullSTTC = None
    num_of_shifts = 500
    # This function performs 500 random circular shifts and calls compute corr for each one of them
    # it returns CtrlGrpMean, CtrlGrpStDev and a random NullSTTC value from the 500
    return CtrlGrpMean, CtrlGrpStDev, NullSTTC

