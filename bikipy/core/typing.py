from pydantic import PositiveInt
from pydantic_numpy import NDArrayFp64

MetersPerPixel = float | NDArrayFp64
TrialId = str | PositiveInt
