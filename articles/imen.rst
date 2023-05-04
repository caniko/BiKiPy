===========================
Post-Hoc kinematic analysis
===========================

Experiments
===========
The coordinates were smoothened with the use of a SARIMAX_ model.


Habituation
-----------
The trial is conducted in an open field, and only motion-related features are measured. The class is, therefore, an alias to :code:`SquareEnclosedTrial`.

Training
--------
The animal is exposed to two identical objects during training. We measure the observation time of both objects defined by the :code:`confinement_filter` followed by the :code:`attention_filter`.

Novelty
-------
One of the objects from the training trial is exchanged with a different object.


Open field
----------
Confinement across video frames in the center can be quantified by defining the center as a rectangular perimeter, and the periphery as the coordinates outside.

The confinement in the rectangular perimeter is tolerance modeled, giving center confinement or CC. We flip the binary values and tolerance model, giving periphery confinement or PC.

CC and PC are binary sequences, where 1 means the coordinate is confined. We compute seconds spent in each zone by taking the sum of the respective sequences and dividing by frames per second.


Methods
=======

Confinement
-----------

Polygon
^^^^^^^
The polygon perimeter includes any triangle, parallelogram (including rectangle), and higher orders polygons. Refer to the source_ for the theory behind the algorithm.

Circle
^^^^^^
Confinement in circle perimeters are computed by (1) computing the distance from the circle center to the provided coordinates:

.. math::
    d_t = | s_t - c |

Where :math:`s_t` is the coordinate of the region of interest in a given time; :math:`c` is the coordinate of the circle center; :math:`d_t` is the distance from the circle center.

Finally, (2) we threshold the distance:

.. math::
    inCircle_t = d_t < r

Where :math:`r` is the radius of the circle.

We do this for all the coordinates in the given trial giving us a sequence of booleans stating the binary confinement of the animal.


Motion
------
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


Tolerance model
---------------
The tolerance model (TM) is a model for binary sequences. The motivation behind this model is the necessity for the elimination of *jitter* from inaccurate measurements. In addition to the binary sequence, TM requires two parameters: (1) minimum seconds of attention (MinSA), and (2) maximum seconds of distraction (MaxSD). The seconds are converted to frames by multiplication with the :code:`fps` (frames per second) value:

.. math::
    MinFA = MSA \cdot fps

    MaxFD = MSA \cdot fps

Where *MinFA* is minimum frames of attention; *MaxFD* maximum frames of distraction.


Algorithm
^^^^^^^^^
MinFA and MaxFD are used to tolerance model the provided binary sequence as follows:

#. :code:`True` must persist for MinFA elements for a tolerated sequence to *start*, and we set the beginning of the sequence to the index of the first :code:`True` value in the sequence.


.. note::
   The entirety of the tolerated sequence will be set to :code:`True`


#. Every :code:`False` will accumulate to a distraction counter till the counter is equal to MaxFD.

#. The tolerance sequence is terminated at the index before the final :code:`False` element.

.. _SARIMAX: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
.. _source: https://github.com/sasamil/PointInPolygon_Py
.. _Euclidean norm: https://en.wikipedia.org/wiki/Euclidean_space#Euclidean_norm
.. _akima: https://en.wikipedia.org/wiki/Akima_spline
.. _finite difference: https://en.wikipedia.org/wiki/Finite_difference
