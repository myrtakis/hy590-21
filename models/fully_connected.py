import tensorflow as tf

from tensorflow.keras.layers import Input, Dense ,Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers,regularizers
from utils.helpers import get_or_default


class FullyConnected():

    def __init__(self):
        self.fc = None

    def build_model(self, setting_configuration, window_config, model_params, input_dim, output_dim):
        
        ### about DNN Configurarations - Fully Connected Layers
        ### number of hidden layers - DNN width 
        ### number of units per layer 
        ### decleration of activation functions across layers
        ### Input Output Dimension
        ### Regularization Strategies
        #### 1)  l1 l2 or both
        #### 2) dropout 
        
        #self.fc = tf.keras.Sequential(        
        #    tf.keras.layers.Dense(units=1)            
        #)
        
        model=tf.keras.Sequential()
        #if window_width >1:
            #tf.Flatten()

        # setup the parameters robustly
        activation = get_or_default(model_params, 'activation', None)
        l1_alpha = float(get_or_default(model_params['regularization'], 'l1', 0))
        l2_alpha = float(get_or_default(model_params['regularization'], 'l2', 0))
        dropout = float(get_or_default(model_params['regularization'], 'dropout', 0))

        #weights=[w,np.random.random(w.shape[1])],
        model.add(tf.keras.Input(shape=(input_dim,)))

        ### add hidden layers
        for units_per_layer in model_params['units_per_hidden_layer']:
            model.add(Dense(units_per_layer,  activation = activation,
                            #weights= [w,0.001*np.random.random(w.shape[1])],
                            activity_regularizer=tf.keras.regularizers.l1_l2(l1=l1_alpha, l2=l2_alpha)))
            model.add(Dropout(dropout))

        model.add(Dense(output_dim, activation = 'linear'))

        self.fc = model
        self.fc.compile(loss=setting_configuration['loss'], metrics=setting_configuration['metrics'],
                            optimizer=setting_configuration['optimizer'])

    def fit_model(self, Xtrain, Xval, epochs=20, callbacks=None):
        history = self.fc.fit(Xtrain, epochs=epochs, validation_data=Xval, callbacks=callbacks)
        return history

    def get_instance(self):
        return self.fc
