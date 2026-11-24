"""Shared by the submission's stages: the instance's files, the access token, the client and the server's timing."""
import base64
import json
import secrets
import sys
import time
from pathlib import Path

from lattica_query import QueryClient, QueryToken
from lattica_query.logging import QUERY_THEME
from lattica_query.transport.backend import BackendAPI
from lattica_query.transport.settings import ClientInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness"))
from params import SINGLE, InstanceParams  # noqa: E402

# The public model the remote worker runs: CryptoFace on 64x64 faces, one pair per query
MODEL_ID = "CryptoFace64"

# Runs reuse one token, so each run replaces its key on the worker instead of adding one.
# A new token is fetched a day before it expires.
TOKEN_PATH = Path(__file__).resolve().parent / ".access_token"
TOKEN_MIN_VALIDITY = 24 * 3600   # seconds


class LocalFilePaths:
    def __init__(self, params: InstanceParams) -> None:
        io_dir = params.iodir()

        self.TEST_PAIRS_PATH    = params.get_test_input_file()
        self.PUBLIC_KEYS_DIR    = io_dir / "public_keys"          # a copy of the evaluation key the worker gets
        self.PREPROCESSED_PATH  = params.io_intermediate_dir() / "pair.pt"
        self.CT_UPLOAD_PATH     = io_dir / "ciphertexts_upload" / "pair.bin"
        self.CT_DOWNLOAD_PATH   = io_dir / "ciphertexts_download" / "score.bin"
        self.SERVER_REPORT_PATH = io_dir / "server_reported.json"
        self.PREDICTIONS_PATH   = params.get_encrypted_model_predictions_file()


def init(argv: list[str]) -> LocalFilePaths:
    """The files of the instance a stage was started for."""
    size = int(argv[1])
    if size != SINGLE:
        raise ValueError(f"The submission serves only SIZE={SINGLE} (a single pair).")
    return LocalFilePaths(InstanceParams(size, rootdir=REPO_ROOT))


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# The token and the client

def _token_expiry(token: str) -> float:
    """The token's expiry time (the JWT's exp claim), in seconds since the epoch."""
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]


def get_demo_token() -> str:
    """A query token for the public model: the one at TOKEN_PATH, unless it expires within TOKEN_MIN_VALIDITY."""
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text()
        if _token_expiry(token) - time.time() > TOKEN_MIN_VALIDITY:
            return token

    with BackendAPI(client_info=ClientInfo.from_package("lattica_query"), theme=QUERY_THEME) as backend:
        response = backend.call("api/token/generate_token_demo", {
            "modelId": MODEL_ID,
            "tokenName": f"face-recognition-{secrets.token_hex(4)}",
        })
    TOKEN_PATH.write_text(response["token"])
    return response["token"]


def get_lattica_client(new_token: bool = False) -> QueryClient:
    """The model's client. Only key generation may take a new token: the later stages need the token of the key."""
    token = get_demo_token() if new_token else TOKEN_PATH.read_text()
    return QueryClient(QueryToken(token))


def server_timing_report(client: QueryClient) -> dict:
    """The server's timing of the client's last query, in seconds. The backend reports
    logic > upload > worker > compute, each including the next; the client returns each one's own share."""
    components = client.query.last_server_timing
    return {
        "Encrypted computation": round(components.get("compute", 0), 4),   # the homomorphic pipeline on the GPU
        "Total": round(sum(components.values()), 4),
        "additional_measurements": {
            "Backend overhead": round(components.get("worker", 0) + components.get("logic", 0), 4),
            "Upload time": round(components.get("upload", 0), 4),          # the query to the worker and the result back
        },
    }
