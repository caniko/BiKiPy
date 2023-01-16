from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Generic, TypeVar, ClassVar

import numpy as np
from pydantic import root_validator, validator
from pydantic.generics import GenericModel
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.base_class import BaseBikipyInspectMixin
from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.feature.physical_object.single.component import AbcObservationComponent
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader
from bikipy.utils.plot.inspect import generic_inspection_finalization

logger = getLogger(__name__)


ObservationComponent = TypeVar("ObservationComponent", bound=AbcObservationComponent)


class PhysicalObject(GenericModel, Generic[ObservationComponent], BaseBikipyInspectMixin, VideoMetadataMixin):
    """
    The physical object is a triadic abstraction of Reader, Perimeter and Trial. This abstraction allows
    us to define methods that require the respective attributes, think of it as a union between the classes!
    """

    label: str
    observation_components: tuple[ObservationComponent, ...]

    # Inspection fields
    trial_obj_label: Optional[TrialId]
    _fig: Any = None
    _axes: Any = None
    _exporting_figure: bool = False

    category = "physical_object"

    @cached_property
    def combined_observation_components(self) -> NDArrayBool:
        result = np.logical_or.reduce((component.combined_sensation for component in self.observation_components))
        self.generic_result_plotter(result, self.summary_axes_row[0], "CombinedComponents")
        return result

    @cached_property
    def post_tolerance_modeled_combined_observation_components(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.combined_observation_components, self.video.fps)
        self.generic_result_plotter(result, self.summary_axes_row[1], "PostToleranceModeledCombinedComponents")
        return result

    @cached_property
    def tolerance_modeled_combined_observation_components(self) -> NDArrayBool:
        result = np.logical_or.reduce(
            (component.tolerance_modeled_combined_sensation for component in self.observation_components)
        )
        self.generic_result_plotter(result, self.summary_axes_row[2], "ToleranceModeledCombinedComponents")
        return result

    @cached_property
    def double_tolerance_modeled_combined_observation_components(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.tolerance_modeled_combined_observation_components, self.video.fps)
        self.generic_result_plotter(result, self.summary_axes_row[3], "DoubleToleranceModeledCombinedComponents")
        return result

    def inspect_attention(self) -> None:
        if isinstance(self.inspect_arg, Path):
            name = f"{self.inspect_arg.stem}_{self.label}.jpg"
            if self.trial_obj_label:
                name = f"{self.trial_obj_label}_{name}"
            generic_inspection_finalization(self.class_inspect_arg / name)
        else:
            generic_inspection_finalization(self.class_inspect_arg)

    @property
    def attention_fig(self):
        if self.fig is not None:
            return self.fig
        self._init_matplotlib()
        return self.fig

    @property
    def attention_axes(self):
        if self.axes is not None:
            return self.axes
        self._init_matplotlib()
        return self.axes

    @property
    def summary_axes_row(self):
        return self.attention_axes[-1]

    @property
    def number_of_components(self) -> int:
        return len(self.observation_components)

    @property
    def max_component_row_length(self) -> int:
        return max(obs_profile.inspection_row_length for obs_profile in self.observation_components)

    def _init_matplotlib(self):
        self.fig, self.axes = self.video.subplots(nrows=self.number_of_components, ncols=self.max_component_row_length)
        self.fig.suptitle(
            "Observation cumulative filtration analysis",
            fontsize=self.video.upscaled_video.plotting_title_font_size * 1.1,
        )

    @cached_property
    def _inspect_pixels(self) -> bool:
        return self.video.frame is not None

    @cached_property
    def _global_attention_kwargs(self) -> dict[str, Any]:
        return {"inspect_video": self.video, "inspect_pixels": self._inspect_pixels}

    @property
    def _first_reader(self) -> Reader:
        return self.observation_components[0].reader

    def generic_result_plotter(self, valid_boolean_index: NDArrayBool, ax: Any, label: str) -> None:
        ax.set_title(label, fontsize=self.video.upscaled_video.plotting_title_font_size)
        ax.scatter(
            *self._first_reader.plot_prepared_kinematic_coordinates[valid_boolean_index].T, marker="x", color="green"
        )
