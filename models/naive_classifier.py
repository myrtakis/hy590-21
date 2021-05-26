import pandas as pd
import numpy as np
import tensorflow as tf


class Naive:

    def __init__(self):
        self.mean_per_neuron = None
        self.loss = None

    def build_model(self, setting_configuration, window_config, model_params, input_dim, output_dim):
        self.loss = setting_configuration['loss']

    def fit_model(self, Xtrain, Xval, epochs=20, callbacks=None):
        Xtrain_as_np = self.__convert_tensor_dataset_to_numpy__(Xtrain)
        # The window already takes care of the lag
        self.mean_per_neuron = pd.DataFrame(Xtrain_as_np).mean().values
        history = {'loss': [], 'val_loss': self.evaluate(Xval)}
        return history

    def get_instance(self):
        return self

    def evaluate(self, data, verbose=None):
        tensor_dataset_as_np = self.__convert_tensor_dataset_to_numpy__(data)
        if self.loss == 'mse':
            return float(np.mean(tf.losses.mean_squared_error(tensor_dataset_as_np, self.mean_per_neuron).numpy()))
        elif self.loss == 'mae':
            return float(np.mean(tf.losses.mean_absolute_error(tensor_dataset_as_np, self.mean_per_neuron).numpy()))
        else:
            print('Uknown loss, None will be returned')
            return None

    @staticmethod
    def __convert_tensor_dataset_to_numpy__(data):
        if isinstance(data, np.ndarray):
            return data
        tesnor_dataset_as_np = list(data.as_numpy_iterator())[0][1]
        # Reduce the tensor to 2-dimensions
        tesnor_dataset_as_np = tesnor_dataset_as_np.reshape(tesnor_dataset_as_np.shape[0],
                                                            tesnor_dataset_as_np.shape[2])
        return tesnor_dataset_as_np