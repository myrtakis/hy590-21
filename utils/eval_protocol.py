import numpy as np


class ForwardChainCV:

    def __init__(self, eval_protocol_config):
        self.mode = eval_protocol_config['mode']
        self.folds = eval_protocol_config['folds']
        self.train_size = eval_protocol_config['train_size']
        self.val_size = eval_protocol_config['val_size']
        self.test_size = eval_protocol_config['test_size']

    def split(self, X):
        X = np.array(X)
        return self.__mode_dispatcher__(X)

    def __mode_dispatcher__(self, X):
        dispatcher = {
            'expanding': self.__expanding__(X),
            'fixed': self.__fixed_sliding__(X),
            'transfer': self.__non_overlapping_transfer_learning__(X),
            'state': self.__non_overlapping_state_wise__(X)
        }
        return dispatcher[self.mode]

    def __expanding__(self, X):
        all_inds = np.arange(0, X.shape[0])
        parts = np.array_split(all_inds, self.folds)
        for i, fold in enumerate(parts):
            fold = np.concatenate(parts[:i+1])
            yield self.__fold_partitioning__(fold)

    def __fixed_sliding__(self, X):
        pass

    def __non_overlapping_transfer_learning__(self, X):
        pass

    def __non_overlapping_state_wise__(self, X):
        pass


    # Helper Functions

    def __fold_partitioning__(self, fold):
        n = fold.shape[0]
        train_inds = fold[0:int(n * self.train_size)]
        val_inds = fold[int(n * self.train_size):int(n * (self.train_size + self.val_size))]
        test_inds = fold[int(n * (1 - self.test_size)):]
        return train_inds, val_inds, test_inds
