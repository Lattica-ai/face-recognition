"""Stage 2: generate the key and register its evaluation key with the worker."""
import shutil
import sys

import submission_utils

paths = submission_utils.init(sys.argv)

with submission_utils.get_lattica_client(new_token=True) as client:
    # Generates the key in the client's key folder and uploads its evaluation key, which the worker preprocesses
    client.keys.generate()
    paths.PUBLIC_KEYS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(client.keys.evaluation_key_path(), paths.PUBLIC_KEYS_DIR)
