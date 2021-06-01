import os
import glob
import matplotlib.pyplot as plt
import json

import numpy as np


def plot_performances(perf_dict, savedir, plot_name):
    for name, perf in perf_dict.items():
        plt.plot(perf, label=name)
    plt.ylabel('performance')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    plt.title(plot_name)
    plt.savefig(savedir + '/' + plot_name + '.png', dpi=300)
    plt.clf()


def naive_compare(naive_dir, model_dir, save_dir):
    assert os.path.isdir(naive_dir), naive_dir + " dir not found"
    assert os.path.isdir(naive_dir), model_dir + " dir not found"

    folds_naive = glob.glob1(naive_dir + '/' + 'train_val_performances', 'fold*')
    folds_model = glob.glob1(model_dir + '/' + 'train_val_performances', 'fold*')
    for fold in set(folds_model).intersection(folds_naive):
        with open(naive_dir + '/' + 'train_val_performances' + '/' + fold) as json_file:
            naive_perfs = json.load(json_file)
        with open(model_dir + '/' + 'train_val_performances' + '/' + fold) as json_file:
            model_perfs = json.load(json_file)
        epochs = len(model_perfs['loss'])
        for k in model_perfs:
            naive_perfs[k] = np.repeat(naive_perfs[k], epochs)
        perf_dict = {'fc mse': model_perfs['loss'], 'fc val mse':model_perfs['val_loss'],
                     'naive mse': naive_perfs['loss'], 'naive val mse': naive_perfs['val_loss']}
        plot_name = fold.split('.')[0] + '_train_val_mse_comparison'
        plot_performances(perf_dict, save_dir, plot_name)
        print(fold)
    pass


def plot_window(model_dir):
    pass
    # import tensorflow as tf
    # WindowGenerator(input_width=wc['input_width'], label_width=wc['label_width'], shift=wc['shift'],
    #                 train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
    #                 settings_config=self.settings_config, input_columns=self.input_column_ids,
    #                 label_columns=self.label_column_ids)
    # new_model = tf.keras.models.load_model('environment/configs/fully_connected/1622130610.438469/'+fold_name+'/model_epoch_005.hdf5')
    # new_model.compile(loss=tf.losses.MeanSquaredError(),
    #      metrics=[tf.metrics.MeanAbsoluteError()])
    #
    # val_performance = {}
    # performance = {}
    # val_performance['new_model'] = new_model.evaluate(window.val)
    # performance['new_model'] = new_model.evaluate(window.test, verbose=0)
    #
    # window.plot(new_model)