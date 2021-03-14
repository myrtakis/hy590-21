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
        neuronA = measurements_df.loc[:, c[0]]
        neuronB = measurements_df.loc[:, c[1]]
        observed_val = compute_corr(neuronA, neuronB, dt)
        CtrlGrpMean, CtrlGrpStDev, NullSTTC = compute_null_sttc(neuronA, neuronB, dt)
        sttc_matrix.append([neuronA.name, neuronB.name, observed_val, CtrlGrpMean, CtrlGrpStDev, NullSTTC])
    return pd.DataFrame(sttc_matrix, columns=col_names)
    

def compute_corr(neuronA, neuronB, dt):
    # This function returns the sttc score
    assert neuronA.shape[0] == neuronB.shape[0]
    
    ### Try for dt =8 20 big sensiticity 
    #neuronA = measurements_df.iloc[:, 0]
    #neuronB = measurements_df.iloc[:, 1]

    neuronA_shifted_plus = neuronA.copy(deep = True)
    neuronB_shifted_minus = neuronB.copy(deep = True)
    ### Find the assembled porpotion of timestamp within +- lag fraction over all spike events per neuron A B respectively 
    for lag in range(dt):
        neuronA_shifted_plus += neuronA_shifted_plus.shift(1, fill_value = 0)
        neuronB_shifted_minus += neuronB_shifted_minus.shift(-1, fill_value = 0)
        
    neuronA_shifted_plus[(neuronA_shifted_plus != 0) & (~(neuronA_shifted_plus.isna()))] = 1
    neuronB_shifted_minus[(neuronB_shifted_minus != 0) & (~(neuronB_shifted_minus.isna()))] = 1
    
    T_A_plus = neuronA_shifted_plus.sum()/neuronA.shape[0]
    T_B_minus = neuronB_shifted_minus.sum()/neuronB.shape[0]
    
    ### Number of splkes found within the lagged interval per direction
    P_A_Bminus = ((neuronA == 1) & (neuronB_shifted_minus == 1)).sum()/neuronA.sum()
    P_B_Aplus = ((neuronB == 1) & (neuronA_shifted_plus == 1)).sum()/neuronB.sum()
    
    section1 = (P_A_Bminus - T_B_minus) / (1 - P_A_Bminus*T_B_minus)
    section2 = (P_B_Aplus - T_A_plus) / (1 - P_B_Aplus*T_A_plus)
    STTC_A_B = 0.5 * (section1 + section2)
    print("STTC A_B",str(STTC_A_B))
    return STTC_A_B


def compute_null_sttc(neuronA, neuronB, dt):        
    # This function performs 500 random circular shifts and calls compute corr for each one of them
    # it returns CtrlGrpMean, CtrlGrpStDev and a random NullSTTC value from the 500
    CtrlGrpMean = None
    CtrlGrpStDev = None
    num_of_shifts = 500
    boostratps_NullSTTC_distribution = []
    np.random.seed(4342)
    for boostratps in range(num_of_shifts):
        
        random_shift_A, random_shift_B = np.random.randint(neuronA.shape[0],None,2)
        ### or keep A fixed and shift B
        neuronA_randomly_shfited = pd.Series(np.roll(neuronA, random_shift_A))
        neuronB_randomly_shfited = pd.Series(np.roll(neuronB, random_shift_B))
        
        boostratps_NullSTTC_distribution.append(compute_corr(neuronA_randomly_shfited, neuronB_randomly_shfited,dt))
    
    
    CtrlGrpMean = np.mean(boostratps_NullSTTC_distribution)
    CtrlGrpStDev = np.std(boostratps_NullSTTC_distribution)
    print(f'Mean {CtrlGrpMean} and Std {CtrlGrpStDev} of Null STTC Distribution')
    return CtrlGrpMean, CtrlGrpStDev, boostratps_NullSTTC_distribution








