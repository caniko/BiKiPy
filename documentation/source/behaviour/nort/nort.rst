=============================
Novel object recognition test
=============================
:code:`bikipy.behaviour.nort` is the submodule that stores the analysis pipeline devised for novel object recognition test.

Class outline
=============
A NORT experiment is a sequence of trials, *habituation*, *training*, *novelty*.

The trials are conducted in a square enclosure; moreover, the NORT classes inherit from :code:`SquareEnclosed` classes. Confinement in the center of the square is defined as bravery, while periphery is defined as fear or cowardice. Each of the three trial classes compute every motion feature.

Habituation
-----------
The trial is conducted in an open field, and only motion-related features are measured. The class is, therefore, an alias to :code:`SquareEnclosedTrial`.

Training
--------
The animal is exposed to two identical objects during training. We measure the observation time of both objects defined by the :code:`confinement_filter` followed by the :code:`attention_filter`.

Novelty
-------
One of the objects from the training trial is exchanged with a different object.
