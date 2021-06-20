Features is a subpackage that stores functions for the computation of behavioural features. These functions can be integrated into :code:`trial`.

.. note:
    Most features are computed on a per video frame basis with the use of :code:`numpy.ndarrays` from the NumPy_ package. The rest of the features are often dependent on the cumulative information, making the use of :code:`numpy.ndarrays` impractical, and have a more pythonic implementation instead; a subset of these are accelerated by Numba_.

Angle
-----
The angle between key parts of the animal body can be used to track functions such as balance and states of focus. Moreover, there are two methods for computing the angle.

:code:`feature.angle.inner_angle` computes the inner angle between two vectors that intersect. The solution is based on the definition of the dot product:

.. math::
    \theta = \cos^{-1} \bigg( \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} \bigg); \quad \theta \in [0, \pi]

:code:`feature.angle.counterclockwise_angel_2d` computes the angle in the counterclockwise direction, using the definition of the determinant and the dot product along with the atan2_ function:

.. math::
    \theta = \pi + \operatorname{atan2} (\det(\mathbf{\hat{b}}, \mathbf{\hat{a}}), \mathbf{\hat{b}} \cdot \mathbf{\hat{a}})

Where :math:`\theta \in [0, 2\pi]`; :math:`\mathbf{\hat{a}}` is in the starting direction; :math:`\mathbf{\hat{b}}` is in the ending direction.


Midpoint
--------
The midpoints are computed with the use of the midpoint between two vectors equation:

.. math::
    midpoint = \frac{\mathbf{b} - \mathbf{a}}{2}


Motion
------
The motion subpackage computes displacement, speed, acceleration, and freeze time. Displacement is computed by the finite derivative of position:
.. math::
    s_{n} = p_{n+1} - p_{n}

Speed:
.. math::
    v_{n} = s_{n+1} - s_{n}

Acceleration:
.. math::
    a_{n} = v_{n+1} - v_{n}

Where :math:`p` is position; :math:`s` is displacement; :math:`v` is speed; :math:`a` is acceleration.

.. _NumPy: https://en.wikipedia.org/wiki/NumPy
.. _Numba: https://en.wikipedia.org/wiki/Numba
.. _atan2: https://en.wikipedia.org/wiki/Atan2