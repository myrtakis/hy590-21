from core.processor import Processor
import pandas as pd
import numpy as np
import json
import os
from filters import data_filters
import argparse

DF_F_FNAME = 'mouse_24705_s3_idx24_df_f.csv'
CELL_MEMBERSHIP_FNAME   = 'mouse24705_cellMembership.csv'
COORDS_FNAME            = 'mouse24705_coords.csv'
MEASUREMENTS_FNAME      = 'mouse24705_IoannisThreshold_1.5dc_full_18min.csv'
BOUNDARIES_FNAME        = 'mouse24705_withinBoundaries_30um.csv'


def read_pipeline_configs(configs_dir):
    with open(configs_dir) as json_file:
        return json.load(json_file)


def read_data(data_dir):
    return {
        'cell_memb_data': pd.read_csv(os.path.join(data_dir, CELL_MEMBERSHIP_FNAME)),
        'coords_data': pd.read_csv(os.path.join(data_dir, COORDS_FNAME)),
        'boundaries_data': pd.read_csv(os.path.join(data_dir, BOUNDARIES_FNAME)),
        # 'measurements_data' : pd.read_csv(os.path.join(data_dir, MEASUREMENTS_FNAME)),
        'df_f_data': pd.read_csv(os.path.join(data_dir, DF_F_FNAME))
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


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confpath', required=True, help='Example: --confpath copnfigs/naive.json')
    parser.add_argument('--datadir', required=True, help='Example: --datadir data')
    parser.add_argument('--savedir', required=False, help='Example: --savedir environment')
    return parser.parse_args()


if __name__ == '__main__':

    args = build_arguments()

    config_name = args.confpath.split('.')[0]

    data_dict = read_data(args.datadir)
    unify_col_ids(data_dict)

    filtered_neuron_ids = filter_neuron_ids(data_dict)

    l23_neurons = data_filters.keep_neurons_of_coords(data_dict['coords_data'], 'z', lambda x: 100 < x < 300)
    l4_neurons = data_filters.keep_neurons_of_coords(data_dict['coords_data'], 'z', lambda x: 300 < x < 500)

    valid_l23_neurons_df_f = data_dict['df_f_data'].iloc[:, list(filtered_neuron_ids.intersection(l23_neurons))]
    valid_l4_neurons_df_f = data_dict['df_f_data'].iloc[:, list(filtered_neuron_ids.intersection(l4_neurons))]

    df_f_valid_data = pd.concat([valid_l4_neurons_df_f, valid_l23_neurons_df_f], axis=1)

    proc = Processor(df_f_valid_data, args.savedir, config_name, read_pipeline_configs(args.confpath),
                     valid_l4_neurons_df_f.columns, valid_l23_neurons_df_f.columns)
    proc.train_evaluate_model()