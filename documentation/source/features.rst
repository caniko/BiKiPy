========
Features
========
Features is a subpackage that stores functions for the computation of behavioural features.

.. note:
    Most features are computed on a per video frame basis with the use of :code:`numpy.ndarrays` from the NumPy_ package. The rest of the features are often dependent on the cumulative information, making the use of :code:`numpy.ndarrays` impractical, and have a more pythonic implementation instead; a subset of these are accelerated by Numba_.


Angle
=====
The angle between key parts of the animal body can be used to track functions such as balance and states of focus. Moreover, there are two methods for computing the angle.

Inner angle
-----------
:code:`feature.angle.inner_angle` computes the inner angle between two vectors that intersect. The solution is based on the definition of the dot product:

.. math::
    \theta = \cos^{-1} \bigg( \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} \bigg) \quad \theta \in [0, \pi]

Clockwise angle
---------------
:code:`feature.angle.counterclockwise_angel_2d` computes the angle in the counterclockwise direction, using the definition of the determinant and the dot product along with the atan2_ function:

.. math::
    \theta = \pi + \operatorname{atan2} (\det(\mathbf{\hat{b}}, \mathbf{\hat{a}}), \mathbf{\hat{b}} \cdot \mathbf{\hat{a}})

Where :math:`\theta \in [0, 2\pi]`; :math:`\mathbf{\hat{a}}` is in the starting direction; :math:`\mathbf{\hat{b}}` is in the ending direction.


Midpoint
========
The midpoints are computed with the use of the midpoint between two vectors equation:

.. math::
    midpoint = \frac{\mathbf{b} - \mathbf{a}}{2}

.. note::
    This module supports

Motion
======
The motion subpackage computes motion related features.

Most applications should use the :code:`features.motion.Motion` class to compute and store these features.

Before computing the displacement, the coordinates have their magnitude or `Euclidean norm`_ computed. Any values that are missing, defined as :code:`np.nan`, are interpolated with the akima_ method. The finite derivative is taken of the prepared displacement data, the resulting data is speed; the finite derivative of speed is acceleration:

.. math::
    s_{n} = p_{n+1} - p_{n} \quad m \in [0, k-1]

Speed:

.. math::
    v_{m} = s_{m+1} - s_{m} \quad m \in [0, n-1]

Acceleration:

.. math::
    a_{l} = v_{l+1} - v_{l} \quad l \in [0, m-1]

Where :math:`p` is position; :math:`s` is displacement; :math:`v` is speed; :math:`a` is acceleration; :math:`k` is the number of frames in the video recording.

The motion class also stores the attribute referred to as :code:`freezing_time`, which is the number of seconds the animal remained immobile; derived from displacement.


Observation
===========
Observation metrics can be computed and analysed with the functions in :code:`features.observation`. These functions are currently used to compute *attention per frame*.

Proximity filter
----------------
The proximity filter

Attention
---------
The attention of the animal is correlated to the time spent observing an object. Moreover, the observation time, and spontaneous distractions should be taken into account.


.. _NumPy: https://en.wikipedia.org/wiki/NumPy
.. _Numba: https://en.wikipedia.org/wiki/Numba
.. _atan2: https://en.wikipedia.org/wiki/Atan2
.. _Euclidean norm: https://en.wikipedia.org/wiki/Euclidean_space#Euclidean_norm
.. _akima: https://en.wikipedia.org/wiki/Akima_spline