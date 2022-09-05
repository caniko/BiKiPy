from pydantic import PositiveInt
from pydantic_numpy import NDArrayFp64

MeterPerPixel = float | NDArrayFp64
TrialId = str | PositiveInt
