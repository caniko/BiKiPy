from logging import getLogger
from pathlib import Path, PurePath
from typing import Any

import cv2
from numpy import ndarray

logger = getLogger(__name__)


def resolve_stem_in_filepath(filepath: Any):
    if filepath is None:
        return

    filepath = Path(filepath).resolve()
    assert filepath.parent.exists(), filepath

    if filepath.exists():
        i = 2
        stem = filepath.stem
        while not filepath.exists():
            filepath.with_name(f"{stem}_{i}.ods")
        logger.warning(
            f"The file exists, and adding index "
            f"increment to the new file, {filepath.stem}"
        )

    return filepath


def read_image(image: Any, imread_flagg: Any = None):
    if isinstance(image, str) or isinstance(image, PurePath):
        image_path = Path(image).resolve()
        assert image_path.exists(), image_path
        image = cv2.imread(str(image_path), flags=imread_flagg)
    else:
        assert isinstance(
            image, ndarray
        ), f"image must be either path or np.ndarray, but got:\n{image}"

    return image
