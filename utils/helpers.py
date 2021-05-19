def get_or_default(data_dict, key, default_val):
    if key not in data_dict:
        return default_val
    else:
        return data_dict[key]