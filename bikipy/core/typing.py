from pydantic import PositiveInt
from pydantic_numpy import NDArray, float64, int16

NDArrayFp64 = NDArray[float64]
NDArrayInt16 = NDArray[int16]
NDArrayUint8 = NDArray
NDArrayBool = NDArray[bool]

TrialId = str | PositiveInt
