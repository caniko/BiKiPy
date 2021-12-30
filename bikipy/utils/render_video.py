# Courtesy of https://github.com/kylemcdonald/python-utils/blob/master/ffmpeg.py
import ffmpeg
import numpy as np
from pydantic import Field, FilePath
from pydantic.dataclasses import dataclass


@dataclass
class VideoWriter:
    filename: FilePath
    video_codec: str = Field(default="hevc_nvenc")
    fps: int = 30
    in_pix_fmt: str = "rgb24"
    out_pix_fmt: str = "yuv420p"
    extra_input_args: dict = Field(default_factory=dict)
    extra_output_args: dict = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process = None

    @property
    def input_args(self):
        return {
            "r": self.fps,
            "framerate": self.fps,
            "pix_fmt": self.in_pix_fmt,
            **self.extra_input_args,
        }

    @property
    def output_args(self):
        return {
            "vcodec": self.video_codec,
            "pix_fmt": self.out_pix_fmt,
            **self.extra_output_args,
        }

    def add(self, frame):
        if not self.process:
            height, width = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(width, height),
                    **self.input_args,
                )
                .filter("crop", "iw-mod(iw,2)", "ih-mod(ih,2)")
                .output(self.filename, **self.output_args)
                .global_args("-loglevel", "quiet")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        conv = frame.astype(np.uint8).tobytes()
        self.process.stdin.write(conv)

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def video_write(filename, images, **kwargs):
    writer = VideoWriter(str(filename), **kwargs)
    for image in images:
        writer.add(image)
    writer.close()
