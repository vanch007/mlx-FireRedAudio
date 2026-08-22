# MLX Native Parity Audit

Audit date: 2026-08-22  
Reference: FireRedTeam/FireRedAudio commit `cbecfb74b55696314d8c888964a3f6aca24ca3bc`

## Verdict

The neural inference path is MLX-native and all six public capability paths pass
real inference on Apple M3 Max. PyTorch/CUDA/MPS/JAX/TensorFlow are absent from the
project runtime. Tokenization, Whisper feature extraction, audio file I/O, resampling,
and conversion of the final MLX waveform to NumPy remain CPU-side boundaries; these
are not model-compute fallbacks.

The official maximum of one-hour audio is structurally supported through 3,000-frame
audio-encoder chunking and long-context fused attention, but a full one-hour stress run
is `pending`. An 87.92-second, six-channel test crossed the chunk boundary and passed.

## Correctness fixes in this audit

- Applied Qwen3.5's required `1 + weight` conversion to zero-centered backbone RMSNorm parameters.
- Restored official RedAE causal attention and removed the extra encoder projection GELU.
- Restored DiT AdaLN branch ordering and affine-free LayerNorm semantics.
- Restored official top-k/top-p sampling and four-beam deterministic ASR.
- Fixed thinking parsing, multi-audio feature insertion, generation role metadata, zero-audio errors, and VAE latent returns.
- Moved ISTFT inverse FFT, Hann window, and overlap-add from NumPy to MLX.
- Added MLX-LM's dedicated Gated Delta Metal kernel and length-adaptive fused attention.

## Verification evidence

Status values: `pass`, `pending`, `missing evidence`.

| Check | Status | Evidence |
|---|---|---|
| Weight mapping | pass | 1,741 model parameters mapped; 0 missing, 0 extra, 0 shape mismatches |
| Runtime dependency audit | pass | No Torch/CUDA/MPS/JAX/TensorFlow imports in project runtime |
| Regression tests | pass | 7/7 unittest cases |
| Python compile check | pass | Package, CLI, benchmark, and tests compile |
| Package build | pass | Source archive and universal wheel built successfully |
| ASR | pass | Correct FLEURS Chinese transcript; balanced benchmark RTF 2.4468 |
| Audio QA + thinking | pass | Correct two-speaker answer and separately parsed reasoning |
| Multi-audio understanding | pass | Two identical inputs correctly answered as identical |
| Zero-shot TTS | pass | 9.92 s output; ASR round-trip recovered target content |
| Acoustic editing | pass | 5.83 s source at speed 0.5 produced 11.84 s output |
| Semantic editing | pass | Rewritten text `花草茶的口味一般，苦一些` plus 3.04 s audio |
| Voice design | pass | Structured timbre description plus 4.96 s audio |
| Long-audio chunk boundary | pass | 87.92 s / 6-channel input produced a correct two-speaker answer |
| Full one-hour stress | pending | Not run due high time and unified-memory cost |

## Measured M3 Max balanced profile

- Hardware: Apple M3 Max, 128 GB unified memory
- MLX: 0.32.1; MLX-LM: 0.31.3
- Flow steps: 4
- Full benchmark wall time: 142.50 s
- Maximum resident set size: 21.66 GB
- Peak memory footprint: 41.05 GB
- No FireRedAudio inference process remained after completion

The BF16 quality profile keeps 10 flow steps. Quantization is not enabled by default:
the audited machine has enough memory, and BF16 preserves checkpoint quality. Run only
one persistent engine unless concurrent copies and their memory cost are intentional.
