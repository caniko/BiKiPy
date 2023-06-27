Experiments
===========
The coordinates from DeepLabCut were smoothened with the use of a SARIMAX_ model.

NORT
----
Novel object recognition test or NORT has three stages: *Habituation*, *training*, *novelty*. The habituation is absent of physical objects, and is only an open field; we record motion and center-periphery confinement features. Training and novelty trials each contain two objects, and we evaluate whether the animal is attentively observing them in every frame by employing qualia heuristics.

The following ignores the hard problem of consciousness, and utilizes heuristics that are inspired from expert annotation agents exclusively.

Qualia heuristics help us automatically define instances of attentive physical object observation. We employ axioms from two categories, ray-casting and proximity, to define qualia heuristics. Ray-casting axioms includes casting rays from a body region and detecting if the rays collide with the physical object. Proximity axioms lets us threshold distances from physical object to the animal, allowing us to filter frames where the animal is too far away from the object for the respective heuristic.

At the discretion of the end-user, heuristics can also be combined using AND or OR logic; the resulting boolean index is treated as an additional result.


Axiom methods
=============

Confinement
-----------
PointInPolygon_Py_ was used to define the confinement of the animal to rectangular perimeters of the objects. The same algorithm is used for the proximity axioms, where we expand the initial perimeter to the maximum distance, which was 6 cm for the NORT experiments.

In line-of-sight
----------------
Solving the iLOS problem for polygons with more than three sides has no general trivial solution. The existing state of the art method is to solve computationally using by casting rays from the origin of interest, and check for collisions with regions of interest.

We need two points of reference to define the direction of the rays. One of the points is always the origin of interest; however, the second point must be defined at the discretion of the designer and is arbitrary.

Considering top-down recordings, the second point for a rodents nose was set to the center of the ears; creating a ray that goes through the snout and exiting through the nose.


Open field
----------
Confinement across video frames in the center can be quantified by defining the center as a rectangular perimeter, and the periphery as the coordinates outside.

The confinement in the rectangular perimeter is tolerance modeled, giving center confinement or CC. We flip the binary values and tolerance model, giving periphery confinement or PC.

CC and PC are binary sequences, where 1 means the coordinate is confined. We compute seconds spent in each zone by taking the sum of the respective sequences and dividing by frames per second.

======
Y-Maze
======
The confinement of the animal needs to be defined before beginning trial-specific analysis. We have three datasets:

#. Dataset, **A**, that defines the location of the animal in every recorded frame. This dataset is in other words bijective with respect to the coordinate dataset.
#. A version of the previous dataset, **B**, that has been reduced with respect to consecutive repeating sequences. The reduction has user-defined tolerance; surjective with respect to the coordinate dataset.
#. Dataset **B** where the centre labels have been excluded, **C**.

Seconds spent in areas
----------------------
.. math::
    t_{a} = n_{a} / fps

Where :math:`t` is time in seconds; :math:`n` is number of occurrences of the respective area label; :math:`fps` is frames per second of the video recording.

The results is stored in a map; key: area, value: seconds.

Area alternations
-----------------
The count of each area label in dataset **B**. The results is stored in a map; key: area, value: alternations.

Sum of triplet alternations
---------------------------
Length of dataset **B** minus 2. The subtraction is used to ignore length of the two initial elements from the reduced sequence, **B**.

Triplet alternation distribution
--------------------------------
A dictionary mapping from area triplet to number of occurrences of the respective triplet as a consecutive sequence in dataset **B**.

Spontaneous alternations
------------------------
The ratio of total number of unique triplet occurrences and, the previously defined, sum of alternations metric. Computed as follows:

.. math::
    SA = n / s

Where :math:`n` is the total number of unique triplets; :math:`s` is the sum of alternations.

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
.. _PointInPolygon_Py: https://github.com/sasamil/PointInPolygon_Py
.. _Euclidean norm: https://en.wikipedia.org/wiki/Euclidean_space#Euclidean_norm
.. _akima: https://en.wikipedia.org/wiki/Akima_spline
.. _finite difference: https://en.wikipedia.org/wiki/Finite_difference
