import pandas as pd
import argparse
import os
from temp_correlation.sttc import sttc
from filters import data_filters
import numpy as np


DATA_DIR = 'data'
calculate_sttc = True

CELL_MEMBERSHIP_FNAME   = 'cellMembershipMouse24705.csv'
COORDS_FNAME            = 'coords.csv'
STTC_PAIRS_FNAME        = 'mouse24705_IoannisThreshold_1.5dc_1st-8min_500-shifts_2-dt_pairs-002.csv'
MEASUREMENTS_FNAME      = 'mouse24705_IoannisThreshold_1.5dc_full_18min.csv'
BOUNDARIES_FNAME        = 'withinBoundaries_30um.csv'


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', action='store', type=str, default=DATA_DIR, required=False, help='Give the directory of the data')
    return parser.parse_args()


if __name__ == '__main__':
    args = build_arguments()

    cell_memb_df = pd.read_csv(os.path.join(args.datadir, CELL_MEMBERSHIP_FNAME))
    coords_df = pd.read_csv(os.path.join(args.datadir, COORDS_FNAME))
    boundaries_df = pd.read_csv(os.path.join(args.datadir, BOUNDARIES_FNAME))
    measurements_df = pd.read_csv(os.path.join(args.datadir, MEASUREMENTS_FNAME))
    # measurements_df = pd.read_csv(os.path.join(args.datadir, 'pair_1_1105.csv'))

    measurements_df.columns = np.arange(measurements_df.shape[1]) + 1    # To have a unified neuron id naming schema
    measurements_df = measurements_df.loc[:3300, :]     # 3300 is the final frame for the 8 minute recording period as required by the task

    neuron_ids = boundaries_df['x']
    neuron_ids = data_filters.keep_neurons_of_area(cell_memb_df[cell_memb_df['neuronID'].apply(lambda x: x in neuron_ids)], 'V1')
    neuron_ids = data_filters.keep_neurons_of_coords(coords_df[coords_df['neuron_id'].apply(lambda x: x in neuron_ids)],
                                                     'z', lambda x: 100 < x < 300) # Keep neurons of L23 layer
    neuron_ids = data_filters.keep_neurons_of_firing_rate(measurements_df.loc[:, neuron_ids], 0.01)
    # Keep the useful neurons
    measurements_df = measurements_df.loc[:, neuron_ids]

    if calculate_sttc:
        sttc_matrix = sttc(measurements_df, dt=2, n_jobs=4, write_dir='results')
