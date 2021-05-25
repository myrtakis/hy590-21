from models.fully_connected import FullyConnected
from models.earlystoppingbylossval import EarlyStoppingByLossVal
from utils.win_gen import *
from utils.eval_protocol import ForwardChainCV
import json
import matplotlib.pyplot as plt
import os

import datetime

class Processor:

    def __init__(self, data_df, config_name, pipeline_configs, input_column_ids, label_column_ids):
        self.data_df = data_df
        self.input_column_ids = input_column_ids
        self.label_column_ids = label_column_ids

        self.filepath = 'environment/' + config_name +"/"+str(datetime.datetime.now().timestamp())
        
        os.makedirs(self.filepath)
        self.save_configs(pipeline_configs)

        # Set configurations from the pipeline_configs instance
        self.settings_config = pipeline_configs['settings']
        self.eval_protocol_config = pipeline_configs['evaluation_protocol']
        self.window_config = pipeline_configs['window']
        self.model_config = pipeline_configs['model']

        # Late-assigned variables
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def __find_model__(self):
        model_dispatcher = {
            'fc': FullyConnected
        }
        return model_dispatcher[self.model_config['name']]()

    def plot_history (self,history, fold_name):
         final_dir = self.filepath + '/plots'
         if not os.path.isdir(final_dir):
             os.makedirs(final_dir)
         plt.plot(history.history['loss'])
         plt.plot(history.history['val_loss'])
         plt.title(self.filepath + ' ' + fold_name)
         plt.ylabel('loss')
         plt.xlabel('epoch')
         plt.legend(['train', 'val'], loc='upper left')
         plt.savefig(final_dir + '/' + fold_name + '.png', dpi=300)
         plt.figure()
         
    def __build_window__(self):
        wc = self.window_config
        return WindowGenerator(input_width=wc['input_width'], label_width=wc['label_width'], shift=wc['shift'],
                               train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                               input_columns=self.input_column_ids, label_columns=self.label_column_ids)
    

    def train_evaluate_model(self):
        fw = ForwardChainCV(self.eval_protocol_config)
        perfomances = {'validation': {}, 'test':{}}
        fold = 1
        for train_inds, val_inds, test_inds in fw.split(self.data_df):
            fold_name = 'fold_' + str(fold)
            print('\n', fold_name)
            self.save_fold_incides(fold_name, train_inds, val_inds, test_inds)
            self.train_df = self.data_df.iloc[train_inds, :]            
            self.val_df = self.data_df.iloc[val_inds, :]
            self.test_df = self.data_df.iloc[test_inds, :]
            
            window = self.__build_window__()
            
            model = self.__find_model__()
            
            model.build_model(setting_configuration=self.settings_config,
                              window_config = self.window_config,
                              model_params = self.model_config['params'],
                              input_dim = window.train.element_spec[0].shape[-1],
                              output_dim = window.train.element_spec[1].shape[-1])

            history = model.fit_model(window.train, window.val, epochs=self.settings_config['epochs'], callbacks = self.get_callbacks(fold_name))
                        
            perfomances['validation'][fold_name] = model.get_instance().evaluate(window.val)
            perfomances['test'][fold_name] = model.get_instance().evaluate(window.test, verbose=0)
            
            ######
            #print("############ Evaluation based on saved models")            
            #new_model = tf.keras.models.load_model('environment/configs/baseline/model_final'+ fold_name + '.ckpt')                                          
            #print(new_model.evaluate(window.val))
            #print(new_model.evaluate(window.test))            
            #####
                                             
            self.plot_history(history, fold_name)

            self.save_fold_incides(fold_name, train_inds, val_inds, test_inds)
            
            fold += 1
        print(perfomances)                

    def get_callbacks(self, fold_name):
        return [
            tf.keras.callbacks.ModelCheckpoint(filepath=self.filepath + '/model_final' + fold_name + '.ckpt',
                                                         save_weights_only=False,
                                                         verbose=1),

            EarlyStoppingByLossVal(['val_loss'], stoppingValue=0.05, file=self.filepath+'/model_earlystopped/'+fold_name)
        ]
    
    def save_fold_incides(self, fold_name, train_inds, val_inds, test_inds):
        inds_dict = {'train': train_inds.tolist(), 'val': val_inds.tolist(), 'test': test_inds.tolist()}
        with open (self.filepath + '/' + fold_name + '_indices.json', 'w') as outfile:
            json.dump(inds_dict, outfile)
            
    def save_configs(self, pipeline_configs):
        with open(self.filepath + '/config.json', 'w') as outfile:
            json.dump(pipeline_configs, outfile)
            
            