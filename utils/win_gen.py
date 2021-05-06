import numpy as np
import tensorflow as tf
import pandas as pd

#### before the windowing step we need to determine ne cross-validation folders 
#### by spiting the time series according to the indexing of the output timeseries
#### paremeterer order of prediction (lag)


class WindowGenerator():

    # input_width: lag
    # label_width:
    # input_columns: its a new Paramater that differentiates the feature domains between input(independent) and output(depedent) variables
    # across the population.
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None, input_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

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
## exempaly data
### Fold <=> data
data = pd.DataFrame(np.random.rand(400,5))

#column_indices = {name: i for i, name in enumerate(data.columns)}

## order of prediction is 1 input width with horizon (shift) := 1
w1 = WindowGenerator(input_width=1, label_width=1, shift=1, train_df = data, val_df= None, test_df = None , label_columns=[3,4], input_columns = [0,1,2])


WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(data[:w1.total_window_size]),
                           np.array(data[100:100+w1.total_window_size]),
                           np.array(data[200:200+w1.total_window_size])])


example_window = tf.stack([np.array(data[:w1.total_window_size])])



example_inputs, example_labels = w1.split_window(features = example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

w1.make_dataset(data)

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = data
#WindowGenerator.val = val
#WindowGenerator.test = test
WindowGenerator.example = example


# Each element is an (inputs, label) pair
w1.train.element_spec

for example_inputs, example_labels in w1.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
