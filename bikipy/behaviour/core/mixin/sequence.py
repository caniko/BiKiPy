from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from bikipy.behaviour.core import Trial


class SequentialExperiment(BaseModel):
    # "Sequence of trial classes designed for the experiment class"
    trial_sequence: ClassVar[tuple["Trial", ...]] = ...
