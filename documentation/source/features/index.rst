========
Features
========
Features is a subpackage that stores functions for the computation of behavioural features.

.. note::
    Most features are computed on a per video frame basis with the use of :code:`numpy.ndarrays` from the NumPy_ package. The rest of the features are often dependent on the cumulative information, making the use of :code:`numpy.ndarrays` impractical, and have a more pythonic implementation instead; a subset of these are accelerated by Numba_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   angle.rst
   attention.rst
   midpoint.rst
   motion.rst


.. _NumPy: https://en.wikipedia.org/wiki/NumPy
.. _Numba: https://en.wikipedia.org/wiki/Numba

