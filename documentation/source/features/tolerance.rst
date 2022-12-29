===============
Tolerance model
===============
The tolerance model (TM) is a model for binary sequences. The motivation behind this model is the necessity for the elimination of *jitter* from inaccurate measurements. In addition to the binary sequence, TM requires two parameters: (1) minimum seconds of attention (MinSA), and (2) maximum seconds of distraction (MaxSD). The seconds are converted to frames by multiplication with the :code:`fps` (frames per second) value:

.. math::
    MinFA = MSA \cdot fps

    MaxFD = MSA \cdot fps

Where *MinFA* is minimum frames of attention; *MaxFD* maximum frames of distraction.


Algorithm
=========
MinFA and MaxFD are used to tolerance model the provided binary sequence as follows:

#. :code:`True` must persist for MinFA elements for a tolerated sequence to *start*, and we set the beginning of the sequence to the index of the first :code:`True` value in the sequence.


.. note::
   The entirety of the tolerated sequence will be set to :code:`True`


#. Every :code:`False` will accumulate to a distraction counter till the counter is equal to MaxFD.

#. The tolerance sequence is terminated at the index before the final :code:`False` element.
