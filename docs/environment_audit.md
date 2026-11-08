# Environment audit and recommendation

Reviewed: 2026-11-08. Original read-only audit retained as historical context.

Current choice: retain pml, validated on CPU and user-terminal CUDA. Exact host
GPU is RTX 5060 Laptop GPU (8,518,041,600 bytes, compute capability 12.0). See
[validation record](environment_validation.md). Earlier unknown-GPU and Python
3.11 recommendations below describe the original audit, not current blockers.

Deadline scope: BigST/PatchSTG on SD 2019, due November 24, 2026. Model-level
forward/backward memory and throughput still need measurement. STAEformer memory
estimates below are historical only. See [roadmap](roadmap.md).

## Observed environment

| Item | Observation |
| --- | --- |
| Platform | Linux x86_64, WSL2 kernel 6.18.33.2-microsoft-standard-WSL2 |
| CPU | AMD Ryzen 7 260, 8 cores / 16 logical CPUs visible |
| RAM | Approximately 11 GiB visible, 9.2 GiB available at inspection |
| Swap | 3 GiB |
| Project volume | Approximately 701 GiB free on the Windows C: mount |
| Temporary volume | Approximately 5.7 GiB; do not use for full dataset staging |
| Conda | 26.7.1; base, RAG, pml, rag environments exist |
| Active shell | base; Python 3.14.7 |
| Existing pml | Python 3.14.8, torch 2.14.1, NumPy 2.5.4 |
| pml PyTorch CUDA build | 13.0 |
| GPU visibility | nvidia-smi: GPU access blocked by operating system; torch reports zero visible devices |
| Git | Clean foundation commit b58e28f observed before documentation edits |
| Remote | git@github.com:emon109848/Traffic-forecasting-current-research.git |

GPU model, VRAM, driver version, and host CUDA support remain unknown. Session
GPU restrictions do not establish that the user's own terminal lacks CUDA.
Installed CUDA toolkit packages do not demonstrate a working driver or kernel.
No CUDA computation, model memory measurement, or training was performed.

## Historical environment proposal — superseded by retaining pml

Keep the existing pml environment unchanged. Prefer a dedicated project Conda
environment with Python 3.11 as a conservative starting candidate for the older
research code and scientific tooling. Select an exact patch release and test the
solver during the next environment milestone. This is an engineering preference,
not a paper requirement or a verified compatibility result.

Use Conda for isolation and Python. An official PyTorch pip wheel inside that
environment is acceptable; record its exact version, CUDA build, and index URL.
PyTorch 2.14.1 is a candidate because it is documented on the official versions
page and installed in pml; this audit has not validated it with these models.
Choose the final wheel after checking the real GPU architecture and driver.
The start-locally page and previous-versions page exposed inconsistent stable
version information during review; do not select a version solely from the
start-page label. See [official versioned installation options](https://pytorch.org/get-started/previous-versions/)
and [local installation guidance](https://pytorch.org/get-started/locally/).

Author environments are reference environments, not a combined requirements file.
For example, PatchSTG lists torch 1.11.0 and older NumPy/pandas versions. Porting
its architecture to a modern stack requires numerical and behavioral checks;
matching architecture alone is not a claim of exact reproduction.
See [PatchSTG author requirements](https://github.com/LMissher/PatchSTG#requirements).

Start with torch, NumPy, and pytest when implementation begins. Add pandas and
PyTables only if the selected HDF source requires them, plus plotting/configuration
packages when used. Do not install graph libraries, torchvision, torchaudio,
experiment services, or timm until there is an actual requirement. Record an
explicit platform-specific environment export after a validated installation.

## Feasibility estimate

A 35,040 x 716 x 1 float32 SD array occupies 100,354,560 bytes (about 95.7 MiB).
This is calculated storage, not measured dataset size or training memory. One
year at five-minute cadence is approximately three times as large for the same
sensors. Raw CA acquisition, HDF loading, copies, metadata, and masks cost more.
Use indexed windows rather than materializing every overlapping sample.

For STAEformer, one explicit FP32 spatial attention-score tensor with B=4,
T=12, heads=4, N=716 costs about 375.5 MiB. This excludes other layers, gradients,
optimizer state, and intermediates; it is not a peak VRAM estimate. Consequently
SD data storage is plausible on visible RAM, but GPU feasibility is unverified.
Start memory checks at batch size 1 and grow after measuring forward/backward.

## Historical request — host environment validation now complete

Run nvidia-smi in the user's own terminal and record GPU name, total VRAM,
driver, and displayed CUDA version. The displayed CUDA version describes driver
capability; it does not select the installed PyTorch build automatically.
Then verify a CUDA tensor operation in the chosen Conda environment. If GPU access
remains restricted only in this agent session, record host-run output separately.

## User-provided host GPU report

On 2026-11-08 the user ran nvidia-smi in their own WSL terminal:

- GPU: NVIDIA GeForce RTX 5060 family; full suffix truncated in output.
- Total VRAM: 8,151 MiB; reported usage 0 MiB at capture.
- NVIDIA-SMI: 615.78.05; KMD: 617.42; CUDA UMD: 13.4.

This confirms host-side GPU visibility despite agent-session restrictions. It
does not yet confirm a successful PyTorch CUDA computation or model feasibility.
The earlier unknown-GPU observations above describe the agent audit, not the host.
Use small initial batches and measure actual forward/backward peak memory;
8 GiB does not guarantee that every model's author settings fit.
