def keep_neurons_of_area(membership_df, area=None):
    if area is None:
        return membership_df['neuronID']
    else:
        return membership_df[membership_df[area] == 1]['neuronID'].values


def keep_neurons_of_firing_rate(measurements_df, frate=None):
    if frate is None:
        return list(measurements_df.columns)
    frame_duration = 0.158
    frate_per_neuron = measurements_df.sum(axis=0) / measurements_df.shape[0] / frame_duration
    return frate_per_neuron[frate_per_neuron > frate].index # Here index is ok, because sum is done per column so the indexes are the IDs


def keep_neurons_of_coords(coords_df, axis=None, coord_func=None):
    if axis is None:
        return coords_df['neuron_id']
    return coords_df[coords_df[axis].apply(coord_func)].loc[:, 'neuron_id']
