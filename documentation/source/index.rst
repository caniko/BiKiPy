====================
BiKiPy documentation
====================

Behavioral kinematics_ Python, or BiKiPy (pronounced like bee-ki-py), is a post-hoc_ data analysis platform.

Motivation
----------
Behavioral experiment analyses doesn't have any standardization and community, which ultimately decreases efficiency. *BiKiPy* is the solution.

Goals
-----
- Function as an information store for the technical implementations of behavioral neuroscience and psychology experiments with kinematics.
- Provide a full stack for kinematic analysis of these experiments.

Overview
--------
TLDR_: :code:`bikipy.reader` is used for ingesting data; :code:`bikipy.behavior` for analysis, the remaining submodules consist of auxiliary functions and classes.

The main benefit of BiKiPy is that every discretely implemented pipeline share a common repository of auxiliary functions such as functions used to determine metrics of motion.

Many of the analytical pipelines have been placed in auxiliary subpackages to ensure accessibility across analysis pipelines that are implemented discretely. These subpackages are defined categorically; for instance, :code:`bikipy.math` stores mathematical code, and :code:`bikipy.utils` stores utility code. The :code:`bikipy.perimeter` submodule, defines classes used for abstracting enclosed areas or perimeters.

reader
++++++
Data can be loaded with an instance of the :code:`reader.base.BaseReader`. :code:`reader.DeepLabCutReader`, inherits from :code:`BaseReader`, can be used for 2D DeepLabCut_ data.

behavior
++++++++
Analysis pipelines for supported experiment designs can be found under :code:`experiment` in their respective submodule. Implementations of these experiments are split into trials and experiments. This structure ensures that trials that belong to one group (experiment), can acquire cross-trial constants directly from its respective experiment object.

Common analytical methods are available in the classes located in :code:`behaviour.base`. These include both experiments and trials.


.. toctree::
   :maxdepth: 3

   behaviour/index
   features/index
   project_structure/index
   project_structure/sequence
   perimeter/index
   annotation
   math


.. _post-hoc: https://en.wikipedia.org/wiki/Post_hoc_analysis
.. _kinematics: https://en.wikipedia.org/wiki/Kinematics
.. _TLDR: https://www.urbandictionary.com/define.php?term=tl%3Bdr
.. _DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
