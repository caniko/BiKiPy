import os

# initialize each DeepLabCutReader object with multiprocessing.
# Useful when initialize approximately 20 or more dlc objects
ENABLE_PROCESS_POOLING = not os.getenv("DISABLE_PROCESS_POOLING", False)
ENABLE_NUMBA = not os.getenv("DISABLE_NUMBA", False)

MATPLOTLIB_SCATTER_ALPHA = 0.50
