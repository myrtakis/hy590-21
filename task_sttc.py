from itertools import combinations

import pandas as pd
import argparse
import os
from temp_correlation.sttc import sttc
from filters import data_filters
import numpy as np
from network.network_metrics import clustering_coef, degree_of_connectivity
from plotting import plotter


DATA_DIR = 'data'

run_sttc = True
compute_cluster_coeff = False
compute_degree_of_con = False

CELL_MEMBERSHIP_FNAME   = 'mouse24705_cellMembership.csv'
COORDS_FNAME            = 'mouse24705_coords.csv'
MEASUREMENTS_FNAME      = 'mouse24705_IoannisThreshold_3nz_1.5dc_full_60min.csv'
BOUNDARIES_FNAME        = 'mouse24705_withinBoundaries_30um.csv'


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


def layer_rule(layer, x):
    if layer == 'l23':
        return 100 < x < 300
    if layer == 'l4':
        return 300 < x < 500


def filter_neuron_ids(data_dict):
    filtered_neuron_ids = [
        data_dict['boundaries_data']['x'].values,
        data_filters.keep_neurons_of_area(data_dict['cell_memb_data'], 'V1'),
        # Next line is bypassed because we did not exclude the neurons of lower firing frequency in neural net
        data_filters.keep_neurons_of_firing_rate(data_dict['measurements_data'], 0.01)
    ]
    final_ids = None
    for array_ids in filtered_neuron_ids:
        final_ids = set(array_ids) if final_ids is None else final_ids.intersection(array_ids)
    return final_ids



def calculate_sttc(layers, reference_layer=None):
    cell_memb_df, coords_df, boundaries_df, measurements_df = read_files()

    measurements_df.columns = np.arange(measurements_df.shape[1]) + 1  # To have a unified neuron id naming schema

    filtered_neuron_ids = filter_neuron_ids({'boundaries_data': boundaries_df, 'cell_memb_data': cell_memb_df, 'measurements_data': measurements_df})

    neurons_of_layers = {}
    for l in layers:
        layer_neurons = data_filters.keep_neurons_of_coords(coords_df, 'z', lambda x: layer_rule(l, x))
        neurons_of_layers[l] = list(filtered_neuron_ids.intersection(layer_neurons))

    other_layers = ''
    neuron_pairs = []
    if reference_layer is not None:
        for ref_neuron in neurons_of_layers[reference_layer]:
            for layer in neurons_of_layers:
                if layer != reference_layer:
                    other_layers += layer + ' '
                    neuron_pairs.extend([(ref_neuron, neuron) for neuron in neurons_of_layers[layer]])
    else:
        for layer in neurons_of_layers:
            neuron_pairs.extend(neurons_of_layers[layer])
        neuron_pairs = list(combinations(list(neuron_pairs), 2))

    col_names = [reference_layer, other_layers, 'STTC'] if reference_layer is not None else None

    return sttc(measurements_df, dt=2, neuron_pairs=neuron_pairs,
                n_jobs=4, write_dir='results', fname='l4_l23_sttc_matrix.csv',
                calculate_null=True, col_names=col_names)


def write_results(df, write_dir):
    if write_dir is not None:
        if not os.path.isdir('results'):
            os.mkdir('results')
        df.to_csv(write_dir, index=False)


if __name__ == '__main__':
    args = build_arguments()

    if run_sttc:
        sttc_matrix = calculate_sttc(layers=['l4', 'l23'], reference_layer='l23')
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