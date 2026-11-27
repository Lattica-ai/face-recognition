"""Client-side preprocessing: a photo to an aligned 64x64 face, and a pair of faces to the model's input."""
import io

import numpy as np
import torch
from PIL import Image

FACE_SIZE = 64
PATCH_SIZE = 32
PATCH_CHANNELS = 16   # the model takes RGB zero-padded to the channels of its first map


def decode_image(encoded) -> np.ndarray:
    """An encoded image of the harness's input store as (H, W, 3) uint8 RGB."""
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_detector():
    """InsightFace's Buffalo-L detector, with one 640x640 pass, on the CPU."""
    from insightface.app import FaceAnalysis

    detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    return detector


def align_face(detector, image: np.ndarray) -> np.ndarray:
    """A photo (H, W, 3 uint8 RGB) as the model's 64x64 face (uint8 RGB): the detector's 5-point alignment
    to 112x112, resized to 64. Without a detected face, the photo's center crop."""
    import cv2
    from insightface.utils.face_align import norm_crop

    bgr = image[:, :, ::-1]
    faces = detector.get(bgr)
    if faces:
        aligned = norm_crop(bgr, faces[0].kps, image_size=112)
        return cv2.resize(aligned, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()

    print("[face_preprocessing] Warning: no face detected, falling back to a center crop", flush=True)
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = image[top:top + side, left:left + side]
    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)


def face_slots(face: np.ndarray) -> torch.Tensor:
    """A 64x64 face (uint8 RGB, HWC) as the slots of its four 32x32 patches, row-major: (4, 2^14).

    A pixel x is x / 127.5 - 1. A patch's channel c at (row, column) is slot (row * 32 + column) * 16 + c,
    with RGB in channels 0-2 and zeros in the other 13."""
    image = torch.from_numpy(np.ascontiguousarray(face)).to(torch.float64) / 127.5 - 1
    grid = FACE_SIZE // PATCH_SIZE
    patches = image.reshape(grid, PATCH_SIZE, grid, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4)
    padded = torch.nn.functional.pad(patches, (0, PATCH_CHANNELS - 3))
    return padded.reshape(grid * grid, PATCH_SIZE * PATCH_SIZE * PATCH_CHANNELS)


def pair_slots(face0: np.ndarray, face1: np.ndarray) -> torch.Tensor:
    """Two faces as one query: (2 faces, 4 patches, 2^14 slots), real. The model's client block pairs them
    before encryption: the first face in the real parts of the slots, the second in the imaginary parts."""
    return torch.stack([face_slots(face0), face_slots(face1)])
