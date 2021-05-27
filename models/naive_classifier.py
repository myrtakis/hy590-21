import pandas as pd
import numpy as np
import tensorflow as tf


class Naive:

    __losses__ = {
        'mse': tf.losses.mean_squared_error,
        'mae': tf.losses.mean_absolute_error
    }

    def __init__(self):
        self.mean_per_neuron = None
        self.loss = None
        self.metric = None

    def build_model(self, setting_configuration, window_config, model_params, input_dim, output_dim):
        self.loss = setting_configuration['loss']
        self.metric = setting_configuration['metrics'][0]

    def fit_model(self, Xtrain, Xval, epochs=20, callbacks=None):
        Xtrain_as_np = self.__convert_tensor_dataset_to_numpy__(Xtrain)
        # The window already takes care of the lag
        self.mean_per_neuron = pd.DataFrame(Xtrain_as_np).mean().values
        train_perf = self.evaluate(Xtrain)
        val_perf = self.evaluate(Xval)
        class history:
            history = {'loss': train_perf[0], 'mae': train_perf[1], 'val_loss': val_perf[0], 'val_mae': val_perf[1]}
        return history

    def get_instance(self):
        return self

    def evaluate(self, data, verbose=None):
        tensor_dataset_as_np = self.__convert_tensor_dataset_to_numpy__(data)
        return [
            float(np.mean(Naive.__losses__[self.loss](tensor_dataset_as_np, self.mean_per_neuron).numpy())),
            float(np.mean(Naive.__losses__[self.metric](tensor_dataset_as_np, self.mean_per_neuron).numpy()))
        ]

    @staticmethod
    def __convert_tensor_dataset_to_numpy__(data):
        if isinstance(data, np.ndarray):
            return data
        tesnor_dataset_as_np = list(data.as_numpy_iterator())[0][1]
        # Reduce the tensor to 2-dimensions
        tesnor_dataset_as_np = tesnor_dataset_as_np.reshape(tesnor_dataset_as_np.shape[0],
                                                            tesnor_dataset_as_np.shape[2])
        return tesnor_dataset_as_np