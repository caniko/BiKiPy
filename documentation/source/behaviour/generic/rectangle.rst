=========
Rectangle
=========

.. _center-periphery-confinement:
Center-Periphery confinement
============================
Confinement across video frames in the center can be quantified by defining the center as a rectangular perimeter, and the periphery as the coordinates outside.

The confinement in the rectangular perimeter is tolerance modeled, giving center confinement or CC. We flip the binary values and tolerance model, giving periphery confinement or PC.

CC and PC are binary sequences, where 1 means the coordinate is confined. We compute seconds spent in each zone by taking the sum of the respective sequences and dividing by frames per second.
