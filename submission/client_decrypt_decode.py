"""Stage 8: decrypt the pair's similarity score."""
import sys

import submission_utils

paths = submission_utils.init(sys.argv)

ct_res = paths.CT_DOWNLOAD_PATH.read_bytes()

with submission_utils.get_lattica_client() as client:
    key = client.keys.load()
    score = client.query.decrypt(ct_res, key=key)

paths.PREDICTIONS_PATH.write_text(f"{score.item():.6f}\n")
