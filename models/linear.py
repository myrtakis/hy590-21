import tensorflow as tf

from tensorflow.keras.layers import Input, Dense ,Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers,regularizers


class FullyConnected():

    def __init__(self):
        self.fc = None

    def build_model(self, setting_configuration, window_config, model_config, input_dim, output_dim):
        
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
                
        params = ([30,40,20],activation, (l1 , l2, drop_out_prob ), task)
        
        #if window_width >1:
            #tf.Flatten()
                    
        #weights=[w,np.random.random(w.shape[1])],
        model.add(tf.keras.Input(shape=(input_dim,)))
        act='sigmoid'
        
        ### add hidden layers
        for layers in i,w in paramrs:          
                act='linear'
            
            model.add(Dense(w.shape[1] ,  activation = act,
                            #weights= [w,0.001*np.random.random(w.shape[1])],
                            activity_regularizer=tf.keras.regularizers.l1_l2(l1=10**(-5),l2=10**(-5))))
             model.add(Dropout(0.10))
        
        model.add()
        self.fc.compile(loss=setting_configuration['loss'], metrics=setting_configuration['metrics'],
                            optimizer=setting_configuration['optimizer'])
        
        
    def fit_model(self, Xtrain, Xval, epochs=20, callbacks=None):
        
        history = self.fc.fit(Xtrain, epochs=epochs, validation_data=Xval)
        
        return history

    def get_instance(self):
        return self.fc
