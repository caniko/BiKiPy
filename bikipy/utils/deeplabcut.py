import numpy as np


def reduce_likelihoods(df, point_names):
    return np.multiply.reduce(
        [df.loc[:, [(point, "likelihood")]].values for point in point_names]
    )


def get_region_of_interest_data(df, region_of_interest):
    return df.loc[:, [(region_of_interest, "x"), (region_of_interest, "y")]].values
