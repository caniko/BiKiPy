import numpy as np
from pydantic import PositiveInt
from pydantic_numpy import NDArrayFp64

MetersPerPixel = float | NDArrayFp64
Label = str | PositiveInt

# uint8 when there are 255 or fewer perimeters in the trial.
ConfinementSequence = np.ndarray[int, np.dtype[np.uint8] | np.dtype[np.uint16]]
