Behavioral [kinematics](https://en.wikipedia.org/wiki/Kinematics) Python, or BiKiPy, is a data analysis platform. The submodules compartmentalize different steps of the analysis. `reader` for ingesting data, `behavior` for analysis. Some of the abstractions have been extracted from the `behavior` submodule, and placed in a categorical submodule (`math` and `utils`) to foster reusability across discrete analysis pipelines. The `perimeter` submodule, defines classes for enclosed areas or perimeters, which is very central to kinematic studies.

## Motivation
Behavioral research doesn't have any centralised methods for standardization, which ultimately slows down the scientific progression of research and industry applications. BiKiPy is the solution.

## Goals
- Function as an information store for the technical implementations of behavioral neuroscience and psychology experiments with kinematics.
- Provide a traditional full stack for the analysis of these experiments.

## reader
Data can be loaded with an instance of the `reader.base.BaseReader`. `reader.DeepLabCutReader`, inherits from `BaseReader`, and provides abstractions for 2D [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) data; state of the art package that can perform markerless tracking (2021).

## behavior
Analysis pipelines for supported experiment designs can be found under `experiment` in their respective module in the `behavior` submodule.
