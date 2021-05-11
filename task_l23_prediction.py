from core.processor import Processor
import pandas as pd
import numpy as np
import json

CONFIGS_PATH = 'configs/test.json'


def read_pipeline_configs():
    with open(CONFIGS_PATH) as json_file:
        return json.load(json_file)


def read_data():
    return pd.DataFrame(np.random.rand(100, 5))


def apply_filters(data):
    pass


def column_ids(data, layer):
    if layer == 'L4':
        return [0, 1, 2]
    else:
        return [3,4]


if __name__ == '__main__':
    data = read_data()
    # data = apply_filters(data)
    proc = Processor(data, read_pipeline_configs(), column_ids(data, 'L4'), column_ids(data, 'L23'))
    proc.train_evaluate_model()
