=============================
Novel object recognition test
=============================
:code:`bikipy.behaviour.nort` is the submodule that stores the analysis pipeline devised for novel object recognition test.

Class outline
=============
A NORT experiment is a sequence of trials, *habituation*, *training*, *novelty*. The trial is conducted in a rectangular box, see section :ref:`center-periphery-confinement` for the overview of measured features in every step. Habituation, being open field, only include the generic features.

Training
--------
The animal is exposed to two identical objects during training. We measure the observation time of both objects defined

Novelty
-------
One of the objects from the training trial is exchanged with a different object.
