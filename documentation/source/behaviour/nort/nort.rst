=============================
Novel object recognition test
=============================
:code:`bikipy.behaviour.nort` is the submodule that stores the analysis pipeline devised for novel object recognition test.

The NORT experiment consists of groups of trials with respect to the animal. Each group consists of *habituation*, *training*, *novel (object exposure)*. NORT trials are conducted in a square enclosure, which is the reason behind the trial objects inheriting from :code:`SquareEnclosedTrial`. This allows us to define a center and perimeter within the square enclosure.

Habituation
===========
There isn't much to say about habituation as it is simply a semantic class. In other words, it is just an alias to :code:`SquareEnclosedTrial`.

Training
========
The animal is exposed to two object during training. In these trials, we are interested in how long the animal observes, furthermore, inspects these objects. We do this by applying a perimeter that segments these objects. At this point we are able to pass the necessary data to :code:`features.attention.polygonal_perimeter_attention` to perform the analysis. As a result, we get the boolean index for attention of these objects.

.. code-block::
    seconds_observed = np.sum(boolean_index) / fps

Each boolean in the :code:`boolean_index` represents one frame.

Novelty
=======
We do the same thing as we did in training; moreover, at this instance, one of the objects (the variable object) has been exchanged with a new object, the novel object.
