from core.processor import Processor
import pandas as pd
import numpy as np
import json
import os
from filters import data_filters

CONFIGS_PATH = 'configs/baseline.json'
DATA_DIR = 'data'

DF_F_FNAME = 'mouse_24705_s3_idx24_df_f-001.csv'
CELL_MEMBERSHIP_FNAME   = 'cellMembershipMouse24705.csv'
COORDS_FNAME            = 'coords.csv'
MEASUREMENTS_FNAME      = 'mouse24705_IoannisThreshold_1.5dc_full_18min.csv'
BOUNDARIES_FNAME        = 'withinBoundaries_30um.csv'


def read_pipeline_configs():
    with open(CONFIGS_PATH) as json_file:
        return json.load(json_file)


def read_data():
    return {
        'cell_memb_data': pd.read_csv(os.path.join(DATA_DIR, CELL_MEMBERSHIP_FNAME)),
        'coords_data': pd.read_csv(os.path.join(DATA_DIR, COORDS_FNAME)),
        'boundaries_data': pd.read_csv(os.path.join(DATA_DIR, BOUNDARIES_FNAME)),
        # 'measurements_data' : pd.read_csv(os.path.join(DATA_DIR, MEASUREMENTS_FNAME)),
        'df_f_data': pd.read_csv(os.path.join(DATA_DIR, DF_F_FNAME))
    }


def unify_col_ids(data_dict):
    # To have a unified neuron id schema
    ids = np.arange(data_dict['df_f_data'].shape[1]) + 1
    data_dict['df_f_data'].columns = ids
    #data_dict['measurements_data'].columns = ids


def filter_neuron_ids(data_dict):
    filtered_neuron_ids = [
        data_dict['boundaries_data']['x'].values,
        data_filters.keep_neurons_of_area(data_dict['cell_memb_data'], 'V1'),
        # TODO ask if we need to keep only the higher frequency neurons in df/f (now it's only for eventograms)
        # data_filters.keep_neurons_of_firing_rate(data_dict['measurements_data'], 0.01)
    ]
    final_ids = None
    for array_ids in filtered_neuron_ids:
        final_ids = set(array_ids) if final_ids is None else final_ids.intersection(array_ids)
    return final_ids


if __name__ == '__main__':

    config_name = CONFIGS_PATH.split('.')[0]

    data_dict = read_data()
    unify_col_ids(data_dict)

    filtered_neuron_ids = filter_neuron_ids(data_dict)

    l23_neurons = data_filters.keep_neurons_of_coords(data_dict['coords_data'], 'z', lambda x: 100 < x < 300)
    l4_neurons = data_filters.keep_neurons_of_coords(data_dict['coords_data'], 'z', lambda x: 300 < x < 500)

    valid_l23_neurons_df_f = data_dict['df_f_data'].iloc[:, list(filtered_neuron_ids.intersection(l23_neurons))]
    valid_l4_neurons_df_f = data_dict['df_f_data'].iloc[:, list(filtered_neuron_ids.intersection(l4_neurons))]

    df_f_valid_data = pd.concat([valid_l4_neurons_df_f, valid_l23_neurons_df_f], axis=1)

    proc = Processor(df_f_valid_data, config_name, read_pipeline_configs(),
                     valid_l4_neurons_df_f.columns, valid_l23_neurons_df_f.columns)
    proc.train_evaluate_model()