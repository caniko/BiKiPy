======
Y-Maze
======
The Y-Maze is defined as a radial maze with three arms. The center of the y-maze is an equilateral triangle, while the arms are parallelograms. Each area is defined as a :code:`bikipy.perimeter`.

.. note::
    As with other radial mazes, the perimeters of the y-maze is most accurately defined by using the :code:`bikipy.perimeter.radial_maze.generate_radial_maze_perimeters` function.

Analysis
========
The temporal perimeter confinement of the animal needs to be defined before beginning trial-specific analysis. We have two datasets:

#. Dataset, **A**, that defines the location of the animal in every recorded frame. This dataset is in other words bijective with respect to the coordinate dataset.
#. A version of the previous dataset, **B**, that has been reduced with respect to consecutive repeating sequences. The reduction has user-defined tolerance. This makes the dataset is surjective with respect to the coordinate dataset.
#. Dataset **B** where the centre labels have been excluded, **C**.

Sum of alternations
===================
Length of dataset **C** minus 2. The subtraction is used to exclude the two initial alternations.

Seconds spent in areas
======================
A dictionary mapping from area label to seconds spent in respective area. The computation of the metric is as follows:

.. math::
    t_{a} = n_{a} / fps

Where :math:`t` is time in seconds; :math:`n` is number of occurrences of the respective area label; :math:`fps` is frames per second of the video recording.

Area alternations
=================
A dictionary mapping from area label to number of alternations into the respective area. This metric is computed by counting the number of occurrences of area label in dataset **B**.

Exploration metric
==================
Exploration is simplified to exploration of a given arm only after exploring the remaining arms. In other words, in dataset **C**, a group of three (total number of arms) consecutive values or triplet must be unique.

Triplet alternation distribution
--------------------------------
A dictionary mapping from area triplet to number of occurrences of the respective triplet as a consecutive sequence in dataset **B**.

Spontaneous alternations
------------------------
The ratio of total number of unique triplet occurrences and, the previously defined, sum of alternations metric. Computed as follows:

.. math::
    SA = n / s

Where :math:`n` is the total number of unique triplets; :math:`s` is the sum of alternations.
