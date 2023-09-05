from pydantic_numpy import NpNDArrayFp64


def ax_imshow_gray(ax, image: NpNDArrayFp64) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
