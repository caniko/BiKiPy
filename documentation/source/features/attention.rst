=========
Attention
=========
Consists primarily of :code:`feature.attention.perimeter_attention` that computes the attentiveness of the animal towards an area defined by a :code:`Perimeter` object. To arrive at attention as a probable qualia_ at a given video frame, certain conditions have to be fulfilled. These conditions are defined as filters for simplicity.

Filters
=======
Note that these filters are designed for rodents, and may be challenging to species with different anatomies. The sequence of explanation is also the sequence of application.

Proximity filter
----------------
The filter has two rules (logical AND) that signify the specimen being close enough to the object for observation:

#. The nose has to be in the vicinity of the object defined by a maximum normal distance. The maximum normal distance is defined by the user.
#. The eye center has to be outside of the confines of the object.

Gaze filter
-----------
The filter makes sure that the :ref:`features/angle:Inner angle` between the gaze vector and the vector of the closes side is less than or equal to a maximum value defined by the user.

Tolerance filter
----------------
The tolerance filter is a temporal filter with two filtering parameters; minimum seconds of attention, and maximum seconds of distraction. The values are converted to number of frame with the :code:`fps` (frames per second) value.

.. _qualia: https://en.wikipedia.org/wiki/Qualia
