from pathlib import Path
from subprocess import Popen, PIPE
import numpy as np

# Typing only
try:
    from matplotlib.figure import Figure
except ImportError:
    pass


def start(*args):
    return Popen(tuple(map(str, args)), stdin=PIPE, stdout=None, stderr=None)


def BGR2YUV(BGR: np.ndarray[tuple[int, int, int]]):
    GRAY = (BGR[:, :, 0] == BGR[:, :, 1]) & (BGR[:, :, 1] == BGR[:, :, 2])
    gray = BGR[GRAY, 0]
    gray[gray == 192] = 183
    BGR = BGR.astype(np.float64)
    B, G, R = BGR[:, :, 0], BGR[:, :, 1], BGR[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 128 + (B - Y) * 0.492
    V = 128 + (R - Y) * 0.877
    # Quantize to 8-bit
    Y = Y.round().clip(0, 255).astype(np.uint8)
    U = U.round().clip(0, 255).astype(np.uint8)
    V = V.round().clip(0, 255).astype(np.uint8)
    # Special treatment: for grayscale pixels, force U and V to 128
    Y[GRAY] = gray
    U[GRAY] = 128
    V[GRAY] = 128
    return np.stack((Y, U, V), axis=0)


def FFMPEG(w: int, h: int, fps: int, outfile: Path):
    if outfile.exists():
        raise FileExistsError(outfile)
    # Start FFmpeg process
    return start(
        "ffmpeg",
        *("-f", "rawvideo"),
        *("-pix_fmt", "yuv444p"),
        *("-s", f"{w}x{h}"),
        *("-r", fps),
        *("-i", "pipe:"),
        *("-c:v", "libx264"),
        *("-pix_fmt", "yuv420p"),
        *("-color_primaries", "bt709"),
        *("-color_trc", "bt709"),
        *("-colorspace", "bt709"),
        *("-color_range", 2),
        *(
            "-bsf:v",
            ":".join(
                (
                    "h264_metadata=video_full_range_flag=1",
                    "colour_primaries=1",
                    "transfer_characteristics=1",
                    "matrix_coefficients=1",
                )
            ),
        ),
        outfile,
    )


class Video:
    proc: Popen | None = None
    frame_shape: tuple[int, int] | None = None

    def __init__(self, outfile: Path, fps: int = 30):
        self.outfile = outfile
        self.fps = fps

    def write(self, img: np.ndarray):
        h, w, *_ = img.shape
        # Check frame shape
        if self.frame_shape is None:
            self.frame_shape = (w, h)
        elif (w, h) != self.frame_shape:
            raise ValueError(f"Frame shape mismatch: {self.frame_shape} != {(w, h)}")
        # Check FFmpeg process
        if self.proc is None:
            self.proc = FFMPEG(w, h, self.fps, self.outfile)
        # Make sure the frame is BGR8
        if img.dtype != np.uint8:
            raise ValueError(f"Unsupported image dtype {img.dtype}")
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.stack((img[:, :, 0],) * 3, axis=-1)
        elif len(img.shape) != 3 or img.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape {img.shape}")
        # Write frame
        self.proc.stdin.write(BGR2YUV(img).tobytes())

    def writeFig(self, fig: Figure, **kwargs):
        from io import BytesIO
        import cv2

        buffer = BytesIO()
        kwargs.update(format="png", dpi=300)
        fig.savefig(buffer, **kwargs)
        buffer.seek(0)
        png = np.frombuffer(buffer.getvalue(), np.uint8)
        img = cv2.imdecode(png, cv2.IMREAD_COLOR)
        return self.write(img)

    def release(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
            self.frame_shape = None

    @classmethod
    def create(cls, outfile: Path | None, *args, **kwargs):
        if outfile is None:
            return None
        else:
            return cls(outfile, *args, **kwargs)
