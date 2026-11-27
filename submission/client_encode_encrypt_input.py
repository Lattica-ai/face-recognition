"""Stage 6: encrypt the pair as one query."""
import sys

import torch

import submission_utils

paths = submission_utils.init(sys.argv)

pair = torch.load(paths.PREPROCESSED_PATH, weights_only=True)

with submission_utils.get_lattica_client() as client:
    key = client.keys.load()
    ct = client.query.encrypt(pair, key=key)
submission_utils.write_bytes(paths.CT_UPLOAD_PATH, ct)
