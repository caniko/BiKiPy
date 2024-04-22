from pydantic_numpy.typing import Np2DArrayFp64


def ax_imshow_gray(ax, image: Np2DArrayFp64) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
