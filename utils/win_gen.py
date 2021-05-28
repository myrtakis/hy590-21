import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#### before the windowing step we need to determine the cross-validation folds
#### by splitting the time series according to the indexing of the output timeseries
#### paremeterer order of prediction (lag)


class WindowGenerator:

    # input_width: The number of input features in the window
    # label_width: The number of output features in the window
    # shift: Lag between input and output features
    # input_columns: It is a new paramater that differentiates the feature domains between input(independent) and output(depedent) variables
    # across the population.
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, settings_config, label_columns=None, input_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.settings_config = settings_config
        
        # Work out the label column indices.
        self.label_columns = label_columns
        self.input_columns = input_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        ##### additional code block
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                          enumerate(input_columns)}
        #####

        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        ##### additional code block
        if self.input_columns is not None:
            inputs = tf.stack([inputs[:, :, self.column_indices[name]] for name in self.input_columns], axis=-1)
        #####

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


    def make_dataset(self, data):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=np.array(data, dtype=np.float32),
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.settings_config['batch_size'] if 'batch_size' in self.settings_config else 32)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            #result = next(iter(self.train))
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result        
        return result
    
    def plot(self, model=None, plot_col = 'T (degC)', max_subplots=1):
        
        
        
        plot_col_input = list(self.input_columns_indices.keys())[1]        
        
        plot_col_label = list(self.label_columns_indices.keys())[1]        
                
        
        inputs, labels = self.example
                
        plt.figure(figsize=(12, 8))
        
        plot_col_index_input = self.column_indices[plot_col_input]
                
        plot_col_index_label = self.column_indices[plot_col_label]
        
        plot_col_index_inputs = [self.input_columns_indices.get(plot_col_input, None) for plot_col_input in self.input_columns_indices.keys()]
            
        
        print(plot_col_index_inputs)
        
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):  
            print("FFF",n)
            plt.subplot(max_n, 1, n+1)
            
            #plt.ylabel(f'{plot_col_index_input} [normed]')
            plt.ylabel( "All Inputs")
            
            plt.plot(self.input_indices, inputs[n, :, plot_col_index_inputs[0]:plot_col_index_inputs[-1]], label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns is not None:
                
                #plot_col_index_label = self.label_columns_indices.get(plot_col_label, None)
                plot_col_index_labels = [self.label_columns_indices.get(plot_col_label, None) for plot_col_label in self.label_columns_indices.keys()]
                print(plot_col_index_labels)
            #else:                
                #plot_col_index_label = plot_col_index
                #if label_col_index is None:
                #    continue
                    
            plt.scatter(self.label_indices, tf.reshape(labels[n, :, plot_col_index_labels[0]:plot_col_index_labels[-1]],-1),
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, plot_col_index_labels[0]:plot_col_index_labels[-1]],
                            marker='X', edgecolors='k', label='Predictions',
                             c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
                
        plt.xlabel('Frames milsecs')
    
    
