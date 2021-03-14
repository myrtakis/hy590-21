def keep_neurons_of_area(membership_df, area=None):
    if area is None:
        return membership_df['neuronID']
    else:
        return membership_df[membership_df[area] == 1].index


def keep_neurons_of_firing_rate(measurements_df, frate=None):
    if frate is None:
        return list(measurements_df.columns)
    # TODO needs completion to filter neuros with firing rate < 0.01 Hz


def keep_neurons_of_coords(coords_df, axis=None, coord_value=None):
    if axis is None and coord_value is None:
        return coords_df['neuron_id']
    # TODO needs completion to filter neurons with 100-300 μm from the z coordinate of cords_df
