import pandas as pd
import argparse
import os
from temp_correlation.sttc import sttc
from filters import data_filters
import numpy as np
from network.network_metrics import clustering_coef, degree_of_connectivity
from plotting import plotter


DATA_DIR = 'data'

run_sttc = False
compute_cluster_coeff = False
compute_degree_of_con = False

CELL_MEMBERSHIP_FNAME   = 'cellMembershipMouse24705.csv'
COORDS_FNAME            = 'coords.csv'
MEASUREMENTS_FNAME      = 'mouse24705_IoannisThreshold_1.5dc_full_18min.csv'
BOUNDARIES_FNAME        = 'withinBoundaries_30um.csv'


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', action='store', type=str, default=DATA_DIR, required=False, help='Give the directory of the data')
    return parser.parse_args()


def read_files():
    cell_memb_df = pd.read_csv(os.path.join(args.datadir, CELL_MEMBERSHIP_FNAME))
    coords_df = pd.read_csv(os.path.join(args.datadir, COORDS_FNAME))
    boundaries_df = pd.read_csv(os.path.join(args.datadir, BOUNDARIES_FNAME))
    measurements_df = pd.read_csv(os.path.join(args.datadir, MEASUREMENTS_FNAME))
    return cell_memb_df, coords_df, boundaries_df, measurements_df


def calculate_sttc():
    cell_memb_df, coords_df, boundaries_df, measurements_df = read_files()

    measurements_df.columns = np.arange(measurements_df.shape[1]) + 1  # To have a unified neuron id naming schema
    # measurements_df = measurements_df.loc[:3300, :]     # 3300 is the final frame for the 8 minute recording period as required by the task

    neuron_ids = boundaries_df['x']
    neuron_ids = data_filters.keep_neurons_of_area(
        cell_memb_df[cell_memb_df['neuronID'].apply(lambda x: x in neuron_ids)], 'V1')
    neuron_ids = data_filters.keep_neurons_of_coords(coords_df[coords_df['neuron_id'].apply(lambda x: x in neuron_ids)],
                                                     'z', lambda x: 100 < x < 300)  # Keep neurons of L23 layer
    neuron_ids = data_filters.keep_neurons_of_firing_rate(measurements_df.loc[:, neuron_ids], 0.01)
    # Keep the useful neurons
    measurements_df = measurements_df.loc[:, neuron_ids]

    return sttc(measurements_df, dt=2, n_jobs=4, write_dir='results')


def write_results(df, write_dir):
    if write_dir is not None:
        if not os.path.isdir('results'):
            os.mkdir('results')
        df.to_csv(write_dir, index=False)


if __name__ == '__main__':
    args = build_arguments()

    if run_sttc:
        sttc_matrix = calculate_sttc()
    else:
        sttc_matrix = pd.read_csv('results/sttc_matrix.csv')

    if compute_cluster_coeff:
        ccoef = clustering_coef(sttc_matrix, sig_threshold=4)
        write_results(ccoef, os.path.join('results', 'clustering_coeff.csv'))
    else:
        ccoef = pd.read_csv('results/clustering_coeff.csv')

    if compute_degree_of_con:
        doc = degree_of_connectivity(sttc_matrix, sig_threshold=4)
        write_results(doc, os.path.join('results', 'degree_of_con.csv'))
    else:
        doc = pd.read_csv('results/degree_of_con.csv')

    #plotter.plot_observed_null_distr(sttc_matrix, sig_threshold=4)
    plotter.plot_zscore(sttc_matrix['zscore'])
    # plotter.plot_ecdf(ccoef['CluCoef'].values, 'Clustering Coefficient')
    # plotter.plot_ecdf(doc['NormDoC'].values, 'Degree of Connectivity')