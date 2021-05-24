# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:41:04 2021

@author: john
"""
from tensorflow.keras.callbacks import Callback


class EarlyStoppingByLossVal(Callback):

    def __init__(self,monitor,stoppingValue,file=None):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.stoppingValue=stoppingValue
        self.file=file
            
    def on_epoch_end(self, epoch, logs={}):
        
        #current = logs.get(self.monitor)        
        #if current is None:
        #warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if all(list(map(lambda x:logs.get(x)<=0.5,self.monitor))):
            self.model.save(self.file+'_mse_loss_'+str(0.5)+'.hdf5')
            
        elif all(list(map(lambda x:logs.get(x)<=0.05,self.monitor))) :
            self.model.save(self.file+'mse_loss_'+str(0.05)+'.hdf5')
            
        elif all(list(map(lambda x:logs.get(x)<=self.stoppingValue,self.monitor))):
            self.model.save(self.file+'mse_loss_'+str(self.stoppingValue)+'.hdf5')            
            self.model.stop_training = True