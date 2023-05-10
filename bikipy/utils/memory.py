from time import sleep

import psutil

MINIMUM_MEMORY_HEADROOM_RATIO = 0.9


def compute_tolerated_available_memory(available_to_headroom_ration: float = MINIMUM_MEMORY_HEADROOM_RATIO) -> float:
    return psutil.virtual_memory().total * (1.0 - available_to_headroom_ration)


def current_memory_headroom(available_to_headroom_ration: float = MINIMUM_MEMORY_HEADROOM_RATIO) -> float:
    return (
        psutil.virtual_memory().available
        - compute_tolerated_available_memory(available_to_headroom_ration)
        - psutil.swap_memory().used
    )


def current_memory_headroom_gb(**kwargs) -> float:
    return current_memory_headroom(**kwargs) / 10**9


def wait_for_more_physical_memory():
    while 0 > current_memory_headroom():
        sleep(1)
