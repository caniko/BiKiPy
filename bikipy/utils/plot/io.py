from pydantic_numpy import NDArrayFp64


def ax_imshow_gray(ax, image: NDArrayFp64) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
