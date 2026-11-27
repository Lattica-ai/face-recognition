"""Stage 5: detect and align the pair's two faces and lay them out as the model's input."""
import sys

import h5py
import torch

import face_preprocessing
import submission_utils

paths = submission_utils.init(sys.argv)

detector = face_preprocessing.load_detector()
with h5py.File(paths.TEST_PAIRS_PATH, "r") as pairs:
    faces = [face_preprocessing.align_face(detector, face_preprocessing.decode_image(pairs[name][0]))
             for name in ("image0", "image1")]

paths.PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(face_preprocessing.pair_slots(*faces), paths.PREPROCESSED_PATH)
