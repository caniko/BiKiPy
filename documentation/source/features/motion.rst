.. _motion:
======
Motion
======
The motion subpackage computes motion related features.

Most applications should use the :code:`features.motion.Motion` class to compute and store these features.

Before computing the displacement, the coordinates have their magnitude or `Euclidean norm`_ computed. Any values that are missing, defined as :code:`numpy.nan` or undefined (**N**ot **A** **N**umber), are either interpolated or separated. This is required

The `finite difference`_ of the prepared displacement data, the resulting data is speed; the finite difference of speed is acceleration:

.. math::
    s_{n} = p_{n+1} - p_{n} \quad m \in [0, k-1]

Speed:

.. math::
    v_{m} = s_{m+1} - s_{m} \quad m \in [0, n-1]

Acceleration:

.. math::
    a_{l} = v_{l+1} - v_{l} \quad l \in [0, m-1]

The total displacement is computed by:

.. math::
    s_{tot} = \sum_{i=0}^{k} p_{i+1} - p_{i}

Where :math:`p` is position; :math:`s` is displacement; :math:`v` is speed; :math:`\mathbf{a}` is acceleration; :math:`k` is the number of frames in the video recording.

The motion class also stores the attribute referred to as :code:`freezing_time`, which is the number of seconds the animal remained immobile; derived from displacement.


.. _Euclidean norm: https://en.wikipedia.org/wiki/Euclidean_space#Euclidean_norm
.. _akima: https://en.wikipedia.org/wiki/Akima_spline
.. _finite difference: https://en.wikipedia.org/wiki/Finite_difference
