.. BiKiPy documentation master file, created by
   sphinx-quickstart on Sun Jun 20 06:37:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BiKiPy's documentation!
==================================

Behavioral kinematics_ Python, or BiKiPy (pronounced like bee-key-py), is a data analysis platform. The submodules compartmentalize different steps of the analysis. :code:`bikipy.reader` for ingesting data, :code:`bikipy.behavior` for analysis.

Some of the analysis functions have been placed in auxiliary subpackages that are defined categorically such as :code:`bikipy.math` and :code:`bikipy.utils`, to ensure accessibility across analysis pipelines that are implemented discretely. The :code:`bikipy.perimeter` submodule, defines classes for enclosed areas or perimeters.

Motivation
----------
Behavioral research doesn't have any centralised repository for standardization, which ultimately slows down the scientific progression of research and industry applications. *BiKiPy* is the solution.

Goals
-----
- Function as an information store for the technical implementations of behavioral neuroscience and psychology experiments with kinematics.
- Provide a full stack for the analysis of these experiments.

reader
------
Data can be loaded with an instance of the :code:`reader.base.BaseReader`. :code:`reader.DeepLabCutReader`, inherits from :code:`BaseReader`, and provides abstractions for 2D DeepLabCut_ data; state-of-the-art package that can perform marker-less tracking (2021).

behavior
--------
Analysis pipelines for supported experiment designs can be found under :code:`experiment` in their respective submodule. Implementations of these experiments are split into trials and experiments. This structure ensures that trials that belong to one group (experiment), can acquire cross-trial constants directly from its respective experiment object.

Common analytical methods are available in the classes located in :code:`behaviour.base`. These include both experiments and trials.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   features
   behaviour/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _kinematics: https://en.wikipedia.org/wiki/Kinematics
.. _DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
