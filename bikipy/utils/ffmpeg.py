# Courtesy of https://github.com/kylemcdonald/python-utils/blob/master/ffmpeg.py
import ffmpeg
import numpy as np


class VideoWriter:
    def __init__(
        self,
        filename,
        video_codec="hevc_nvenc",
        fps=30,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.filename = filename
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["r"] = self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = video_codec

    def add(self, frame):
        if self.process is None:
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
