======
Y-Maze
======
The Y-Maze is defined as a radial maze with three arms. The center of the y-maze is an equilateral triangle, while the arms are parallelograms. Each area is defined as a :code:`bikipy.perimeter`.

Analytics
=========
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
