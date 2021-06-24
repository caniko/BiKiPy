========
Features
========
Features is a subpackage that stores functions for the computation of behavioural features.

.. note::
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
    This module supports direct integration into readers that store their primary data as :code:`pandas.DataFrame`, see :code:`feature.midpoint.compute_from_dlc_df`.

Motion
======
The motion subpackage computes motion related features.

Most applications should use the :code:`features.motion.Motion` class to compute and store these features.

Before computing the displacement, the coordinates have their magnitude or `Euclidean norm`_ computed. Any values that are missing, defined as :code:`np.nan`, are interpolated with the akima_ method. The `finite difference`_ of the prepared displacement data, the resulting data is speed; the finite difference of speed is acceleration:

.. math::
    s_{n} = p_{n+1} - p_{n} \quad m \in [0, k-1]

Speed:

.. math::
    v_{m} = s_{m+1} - s_{m} \quad m \in [0, n-1]

Acceleration:

.. math::
    a_{l} = v_{l+1} - v_{l} \quad l \in [0, m-1]

Where :math:`p` is position; :math:`s` is displacement; :math:`v` is speed; :math:`\mathbf{a}` is acceleration; :math:`k` is the number of frames in the video recording.

The motion class also stores the attribute referred to as :code:`freezing_time`, which is the number of seconds the animal remained immobile; derived from displacement.


Attention
=========
Consists primarily of :code:`feature.attention.polygonal_perimeter_attention` that computes the attentiveness of the animal with respect to a :code:`PolygonalPerimeter`. To arrive at attention as a *probable* qualia_ at a given video frame, certain conditions need to be met:

#. The nose has to be within the vicinity of the object, while the center_of_mass has to be outside of the confines of the object. The vicinity is defined by a secondary :code:`PolygonalPerimeter` generated with the :code:`PolygonalPerimeter.border` method.
#. The gaze direction, vector from eye center to nose in rodents for instance, has to be directed at the respective :code:`PolygonalPerimeter`
#. The two previous conditions need to occur for a given amount of time with some tolerance for distraction.


.. _NumPy: https://en.wikipedia.org/wiki/NumPy
.. _Numba: https://en.wikipedia.org/wiki/Numba
.. _atan2: https://en.wikipedia.org/wiki/Atan2
.. _Euclidean norm: https://en.wikipedia.org/wiki/Euclidean_space#Euclidean_norm
.. _akima: https://en.wikipedia.org/wiki/Akima_spline
.. _finite difference: https://en.wikipedia.org/wiki/Finite_difference
.. _qualia: https://en.wikipedia.org/wiki/Qualia