# Lattica Face-Verification Submission

This is Lattica's submission to the face-verification workload of the FHE benchmarking suite. It runs
[CryptoFace](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
(the benchmark's checkpoint, [`halmsu/cryptoface-v1`](https://huggingface.co/halmsu/cryptoface-v1), for 64x64
faces) under RNS-CKKS on a remote Lattica worker with a GPU. The client is this repository's code with the
[`lattica-query`](https://pypi.org/project/lattica-query/) package; the model is deployed on the worker as the
public model `CryptoFace64`.

The submission serves the single-pair instance (size `0`): one query is one pair of faces.

---

## Stages

| Stage | Script | What it does |
|------:|--------|--------------|
| 2 | `client_key_generation` | Takes an access token for the model, generates the key locally and uploads its evaluation key, which the worker preprocesses. |
| 3 | `server_preprocess_model` | Nothing: the model is already deployed on the worker. |
| 5 | `client_preprocess_input` | Detects and aligns the two faces (InsightFace Buffalo-L, one 640x640 pass, 5-point alignment to 112x112, resized to 64x64) and lays them out as the model's input. |
| 6 | `client_encode_encrypt_input` | Encrypts the pair as one query of four ciphertexts. |
| 7 | `server_encrypted_compute` | Sends the query to the worker, which computes the encrypted score, and saves the worker's timing. |
| 8 | `client_decrypt_decode` | Decrypts the score. |
| 9 | `client_postprocess` | Nothing. |

The secret key never leaves the client: the worker gets the evaluation key, the encrypted query and nothing
else. The worker returns one ciphertext, and the similarity score is the only value the client decrypts.

---

## The circuit

The client cuts each aligned face into its four 32x32 patches and sends four ciphertexts, one per patch
position: the first face's patch in the real parts of the complex slots, the second's in the imaginary parts.

The worker runs CryptoFace's patch network (a small ResNet with HerPN activations, a per-channel quadratic)
at each position, on both faces at once. The weights are real, so a convolution acts on both faces of a
ciphertext; the two are taken apart only where the circuit multiplies them or bootstraps them. It then sums
the four patch features of each face into its 256-d embedding and returns the cosine of the two embeddings,
each normalized by a quadratic approximation of the inverse norm in the embedding's squared norm.

A face's path takes 27 levels, 23 for a patch and 4 for the cosine, with 9 levels between bootstraps. The four
positions share their bootstraps, so a query takes 3 of them.

---

## Cryptographic parameters and security

* **Ring dimension:** 2**16 = 65536, with 2**14 slots per patch
* **Levels:** 30-bit primes, nine per bootstrap; plaintext scale 2**30
* **Total modulus size:** ~1557 bits (1008 for the levels and the bootstrap, 549 for nine 61-bit key-switching primes)
* **Secret key distribution:** sparse ternary, Hamming weight 1024

The parameters target 128 bits of security. The estimate can be reproduced with the
[lattice estimator](https://github.com/malb/lattice-estimator):
```python
from estimator import *
Logging.set_level(Logging.LEVEL0)
params = LWE.Parameters(n=2**16, q=2**1557, Xs=ND.SparseTernary(512), Xe=ND.DiscreteGaussian(3))
LWE.estimate.rough(params)
```

---

## Running the submission

Install the dependencies as the top-level README describes, then from the repository root:

```console
python3 harness/run_submission.py 0 --num_runs 3
```

Notes on what the harness reports:

* **Server Reported.** `Encrypted computation` is the worker's own time for the homomorphic pipeline on the GPU.
  The harness's `Encrypted computation` stage is the whole round trip: the query's upload, the backend and the
  result's download included.
* **Key Generation** includes the evaluation key's upload and the worker's preprocessing of it.
* **Packed model weights** reads zero: the model's weights are on the worker, not in `io/server_data`.
* **Public and evaluation keys** is the evaluation key, the part of the key the worker gets. The client keeps
  the key itself in `lattica-query`'s key folder.

The access token is cached in `submission/.access_token` and reused, so each run replaces its key on the
worker. It is a credential: do not commit it.
