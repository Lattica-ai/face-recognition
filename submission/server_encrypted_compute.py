"""Stage 7: run the query on the remote worker."""
import json
import sys

import submission_utils

paths = submission_utils.init(sys.argv)

ct = paths.CT_UPLOAD_PATH.read_bytes()

with submission_utils.get_lattica_client() as client:
    ct_res = client.query.execute(ct)
    server_report = submission_utils.server_timing_report(client)

submission_utils.write_bytes(paths.CT_DOWNLOAD_PATH, ct_res)
paths.SERVER_REPORT_PATH.write_text(json.dumps(server_report))
