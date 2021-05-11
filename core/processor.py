from models.linear import Linear
from utils.win_gen import *
from utils.eval_protocol import ForwardChainCV


class Processor:

    def __init__(self, data_df, pipeline_configs, input_column_ids, label_column_ids):
        self.data_df = data_df
        self.input_column_ids = input_column_ids
        self.label_column_ids = label_column_ids

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
            'linear': Linear
        }
        return model_dispatcher[self.model_config['name']]()

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
            fold_name = 'Fold ' + str(fold)
            print('\n', fold_name)
            self.train_df = self.data_df.iloc[train_inds, :]
            self.val_df = self.data_df.iloc[val_inds, :]
            self.test_df = self.data_df.iloc[test_inds, :]
            window = self.__build_window__()
            model = self.__find_model__()
            model.build_model(setting_configuration=self.settings_config)
            model.fit_model(window.train, window.val)
            perfomances['validation'][fold_name] = model.get_instance().evaluate(window.val)
            perfomances['test'][fold_name] = model.get_instance().evaluate(window.test, verbose=0)
            fold += 1
        print(perfomances)