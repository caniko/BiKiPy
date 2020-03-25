import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def clockwise_angle(vector_1, vector_2):
    """ Returns the inner_angle in radians between vectors 'vector_1' and 'vector_2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    vector_1_u = np.apply_along_axis(unit_vector, 1, vector_1)
    vector_2_u = np.apply_along_axis(unit_vector, 1, vector_2)
    
    inner_angle = np.arccos(np.clip(np.einsum('ij,ij->i', vector_1_u, vector_2_u), -1.0, 1.0))

    # Find determinant
    determinants = np.zeros((len(vector_1_u), 1))
    for i in range(len(vector_1_u)):
        determinants[i] = np.linalg.det(np.vstack((vector_1_u[i], vector_2_u[i])))

    clockwise_angle = np.where(determinants.T >= 0, inner_angle, 2*np.pi - inner_angle)[0]

    # The degree range is [pi, -pi]
    clockwise_angle -= np.pi

    return clockwise_angle


def angle_over_time(dlca, point_a, point_b, point_c):
    df = dlca.df

    position = {}
    for point in (point_a, point_b, point_c):
        position[point] = df.loc[:, [(point, 'x'), (point, 'y')]].values

    ab_vec = position[point_a] - position[point_b]
    bc_vec = position[point_b] - position[point_c]

    angles = clockwise_angle(ab_vec, bc_vec)
    accuracy_score = (df.loc[:, [(point_a, 'likelihood')]].values
                      + df.loc[:, [(point_b, 'likelihood')]].values
                      + df.loc[:, [(point_c, 'likelihood')]].values) / 3
    
    return {'Angles': angles, 'Accuracy Score': accuracy_score.T[0]}
