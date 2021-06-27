
Welcome to BiKiPy's documentation!
==================================
Behavioral kinematics_ Python, or BiKiPy (pronounced like bee-key-py), is a data analysis platform.

Motivation
----------
Behavioral research doesn't have any centralised repository for standardization, which ultimately slows down the scientific progression of research and industry applications. *BiKiPy* is the solution.

Goals
-----
- Function as an information store for the technical implementations of behavioral neuroscience and psychology experiments with kinematics.
- Provide a full stack for the analysis of these experiments.

Overview
--------
TLDR_: :code:`bikipy.reader` is used for ingesting data; :code:`bikipy.behavior` for analysis, the remaining submodules consist of auxiliary functions and classes.

The main benefit of BiKiPy is that every discretely implemented pipeline share a common repository of auxiliary functions.

Many of the analytical pipelines have been placed in auxiliary subpackages that are defined categorically such as :code:`bikipy.math` and :code:`bikipy.utils`, to ensure accessibility across analysis pipelines that are implemented discretely. The :code:`bikipy.perimeter` submodule, defines classes for enclosed areas or perimeters.

reader
++++++
Data can be loaded with an instance of the :code:`reader.base.BaseReader`. :code:`reader.DeepLabCutReader`, inherits from :code:`BaseReader`, and provides abstractions for 2D DeepLabCut_ data; state-of-the-art package that can perform marker-less tracking (2021).

behavior
++++++++
Analysis pipelines for supported experiment designs can be found under :code:`experiment` in their respective submodule. Implementations of these experiments are split into trials and experiments. This structure ensures that trials that belong to one group (experiment), can acquire cross-trial constants directly from its respective experiment object.

Common analytical methods are available in the classes located in :code:`behaviour.base`. These include both experiments and trials.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   behaviour/index.rst
   features/index.rst
   annotation.rst
   math.rst


.. _kinematics: https://en.wikipedia.org/wiki/Kinematics
.. _TLDR: https://www.urbandictionary.com/define.php?term=tl%3Bdr
.. _DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
