# Lattica Face-Verification Submission

This is Lattica's submission to the face-verification workload of the FHE benchmarking suite. It runs
[CryptoFace](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
(the benchmark's checkpoint, [`halmsu/cryptoface-v1`](https://huggingface.co/halmsu/cryptoface-v1), for 64x64
faces) under RNS-CKKS on a remote Lattica worker with a GPU. The client is this repository's code with the
[`lattica-query`](https://pypi.org/project/lattica-query/) package.

This document describes the workload structure, the execution environment, the cryptographic parameters, the
circuit with its level management, and how to run the submission.

---

## Workload configuration

* **Mode:** SINGLE (size `0`)
* **Query:** one pair of faces
* **Model input:** 64x64 aligned faces, as four 32x32 patches each
* **Result:** one similarity score, the cosine of the two faces' 256-d embeddings

The submission serves the single-pair instance only.

---

## Execution environment

**Client**

* AWS EC2 c8a.8xlarge, in the server's region (us-east-1)
* AMD EPYC 9R45: 32 vCPUs, 64 GB RAM

**Server**

* AWS EC2 g7e.2xlarge
* One NVIDIA GB202 GPU (96 GB)
* 8 vCPUs, 64 GB RAM

The client reaches the server through Lattica's public API, so key generation, which uploads the evaluation key,
and the harness's encrypted computation stage include that network.

---

## Cryptographic parameters and security

The submission uses the following parameters:

* **Ring dimension:** 2**16 = 65536
* **Total modulus size:** ~1557 bits (270 compute levels + 738 bootstrapping + 549 GHS)
* **Secret key distribution:** sparse ternary, Hamming weight 1024

With this secret key distribution, the ring dimension supports a modulus of up to ~1721 bits at the 128-bit
security target. The ~1557-bit modulus used here is below that bound, so the parameters are at least 128-bit
secure.

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
ciphertext; the two are taken apart only where the circuit multiplies them: a HerPN's square and the cosine.
A bootstrap takes the two faces as one complex ciphertext. The worker then sums the four patch features of
each face into its 256-d embedding and returns the cosine of the two embeddings, each normalized by a
quadratic approximation of the inverse norm in the embedding's squared norm.

---

## Pipeline levels breakdown

### 2D moduli chain

We are using a technique that we call a *2D modulus chain*, which allows us to decouple optimizations related to
plaintext scaling via mod-switching, and optimizations related to the machine register size.

Each ROW of our 2D chain is a single int64 number, which is the product of one or more "good" primes.
Conceptually, each element in the 2D chain can be consumed as a level (in the context of mod-switching), so the
overall number of levels is the sum of the number of COLs in each ROW. However, the size and computational
overhead grows with the number of ROWS rather than with the number of levels.

### Moduli chain structure
```python
full_q_list_precision=(
                    (30,),     # one level
                    (60, 30),  # two levels
                    (60, 30),
                    (60, 30),
                    (60, 30),
                ),             # nine 30-bit levels in five rows, restored by each bootstrap
q0_precision        = 53       # plus q0, the bootstrap's base prime, under these rows: a fresh ciphertext has 6 limbs
ghs_keyswtich_scale = 9*(61,)  # scale-up by additional 549 bits

input_pt_scale    = 2**30
n_slots           = 2**14      # per ciphertext: a 32x32 patch with 16 channels
```

### Scale management and level consumption

* Every level is a 30-bit prime: a convolution or a multiplication by a constant takes one, and a HerPN's
  square takes one.
* A face's path takes 27 levels: 23 for a patch (the stem, five residual blocks of 4, the pooling HerPN and
  the head) and 4 for the cosine. That is three times the 9 levels a bootstrap restores.
* The patches bootstrap at two points of the network, and each segment spends all its levels: a bootstrap
  gets a ciphertext on one limb, and the score comes out on the last limb alone.
* The four patch positions share their bootstraps. The convolutions keep the gap a stride opens, so two
  positions' maps, and later all four, fit one ciphertext and bootstrap together: a query takes 3 bootstraps
  where one per position and point would take 8.

---

## Performance characteristics

* Evaluation key: 4.4 GB
* Encrypted query: 12.0 MB
* Encrypted result: 1.0 MB
* Total inference time: 1.10 - 1.16 s (**includes network time**, of which 0.19 - 0.21 s on the backend)
* Compute inference time: 154 ms (**compute time only**)

### Homomorphic computation time breakdown

* Convolutions: 112.3 ms
* Bootstraps: 24.1 ms
* HerPN activations and residual additions: 12.8 ms
* Embedding sum and cosine: 2.1 ms
* Packing positions for the bootstraps: 1.5 ms
* Average pooling: 1.3 ms

---

## Comparative reference

Relative to the reference Orion implementation provided with the benchmark suite, as measured in the upstream
suite's `measurements/single`:

| **Stage** | **Lattica** | **Orion** |
| --- | --- | --- |
| **Key Generation** | 64.8 s (10.5 s to generate, the rest mostly the key's upload) | 288.7 s |
| **Encrypted model preprocessing** | 0 s (the model is on the worker) | 1449.7 s |
| **Input encryption** | 1.19 - 1.20 s | 66.1 - 85.5 s |
| **Encrypted computation** | 0.154 s (1.10 - 1.16 s end-to-end) | 107.9 - 111.2 s (733.8 - 749.9 s end-to-end) |
| **Result decryption** | 1.11 - 1.12 s | 31.5 - 32.3 s |
| **Public & evaluation keys** | 4.4 G | 28.8 G |
| **Packed model weights** | 25.9 G (on the worker's GPU) | 114.1 G |
| **Encrypted input** | 12.0 M | 128.1 M |

* Encrypted computation is approximately **700x faster**, and about **650x faster** end-to-end.
* The keys, the packed model weights and the encrypted input are **6.5x, 4.4x and 10.7x smaller**, respectively.

---

## Running the submission

Install the dependencies as the top-level README describes, then run the single pair from the repository root.
The root `requirements.txt` carries `lattica-query` in place of the reference submission's Orion, so the standard
install is all the submission needs:

```console
bash scripts/install_system_deps.sh
bash scripts/install_python_deps.sh
source .venv/bin/activate
python3 harness/run_submission.py 0 --num_runs 3
```

Notes on what the harness reports:

* **Server Reported.** `Encrypted computation` is the worker's own time for the homomorphic pipeline on the GPU,
  `Backend overhead` the backend's own time around it, and `Upload time` the query's way to the worker and the
  result's way back. The harness's `Encrypted computation` stage is the whole round trip: the query's upload,
  the backend and the result's download included.
* **Key Generation** includes the evaluation key's upload and the worker's preprocessing of it. Of the 64.8 s,
  about 10.5 s generate the key, 41 s upload its 4.4 GB evaluation key, 8 s are the worker's preprocessing, and
  the remaining 5 s fetch the model's initialization data and copy the key for the harness.
* **Packed model weights** reads zero, with a harness warning that `io/server_data` does not exist: the model is
  compiled on Lattica's backend and stays on the server. Loaded on the worker's GPU, the packed plaintexts take
  25.9 GiB.
* **Public and evaluation keys** is the evaluation key, the part of the key the worker gets. The client keeps
  the key itself in `lattica-query`'s key folder.
