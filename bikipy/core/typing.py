import numpy as np
from pydantic import PositiveInt
from pydantic_numpy.typing import Np2DArrayFp64

MetersPerPixel = float | Np2DArrayFp64
Label = str | PositiveInt

# uint8 when there are 255 or fewer perimeters in the trial.
ConfinementSequence = np.ndarray[int, np.dtype[np.uint8] | np.dtype[np.uint16]]
