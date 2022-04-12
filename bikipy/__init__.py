import os

# initialize each DeepLabCutReader object with multiprocessing.
# Useful when initialize approximately 20 or more dlc objects
ENABLE_PROCESS_POOLING = os.getenv("ENABLE_PROCESS_POOLING", False)
