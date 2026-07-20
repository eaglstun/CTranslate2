> ⚠️ **Experimental fork — built with AI.** This fork adds a new Apple Silicon **Metal GPU backend** (FP32, FP16, and a native INT8 path) that upstream CTranslate2 does not have. The Metal code was written with **Claude Opus** and **Claude Fable** (Anthropic). It is research-grade: its output is checked against the CPU version, but it makes no promises about stability or backward compatibility. If you need something for production, use [upstream CTranslate2](https://github.com/OpenNMT/CTranslate2). For the design and current status, see [`METAL_BACKEND.md`](METAL_BACKEND.md).

# CTranslate2

CTranslate2 is a C++ and Python library for fast inference with Transformer models.

It runs models on its own custom engine. The engine uses tricks like weight quantization, layer fusion, and batch reordering to [run Transformer models faster and with less memory](#benchmarks), on both CPU and GPU.

These model types are supported:

- Encoder-decoder models: Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper, T5Gemma, T5Gemma2, MADLAD-400
- Decoder-only models: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, Gemma, CodeGen, GPTBigCode, Falcon, Qwen2
- Encoder-only models: BERT, DistilBERT, XLM-RoBERTa

Before you can run a model, you convert it to an optimized format. The library ships converters for several frameworks:

- [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
- [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
- [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
- [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
- [OPUS-MT](https://opennmt.net/CTranslate2/guides/opus_mt.html)
- [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

The project is built for production and promises [backward compatibility](https://opennmt.net/CTranslate2/versioning.html). It also includes some experimental features for shrinking models and speeding up inference.

## Key features

- **Fast, lightweight execution on CPU and GPU** — On supported models, it [runs faster and uses fewer resources](#benchmarks) than general-purpose deep learning frameworks. This comes from many optimizations: layer fusion, padding removal, batch reordering, in-place operations, and caching.
- **Quantization and reduced precision** — Models can store and compute weights at [lower precision](https://opennmt.net/CTranslate2/quantization.html): 16-bit float (FP16), 16-bit brain float (BF16), 16-bit integer (INT16), 8-bit integer (INT8), and 4-bit AWQ (INT4).
- **Many CPU architectures supported** — It runs on x86-64 and ARM64 (AArch64) processors, and can use several tuned backends: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
- **Apple Silicon GPU support (experimental)** — An [Apple Metal](https://developer.apple.com/metal/) backend (`-DWITH_METAL=ON`) runs models on Apple Silicon GPUs in FP32, FP16, and INT8. A full encoder-decoder runs end-to-end on the GPU, and its output matches the CPU. It now beats the Accelerate CPU backend in every measured regime — GEMM-heavy prefill, autoregressive decode, and Whisper beam-search decode (large-v3 FP16 ~4× faster than CPU, transcript byte-identical). See [`METAL_BACKEND.md`](METAL_BACKEND.md) for status and [`METAL_BENCHMARKS.md`](METAL_BENCHMARKS.md) for benchmarks.
- **Automatic CPU detection** — One binary can hold several backends (like Intel MKL and oneDNN) and instruction sets (like AVX, AVX2). It picks the right one at runtime based on your CPU.
- **Parallel and async execution** — You can run many batches at once across multiple GPUs or CPU cores.
- **Dynamic memory usage** — Memory grows and shrinks with the size of each request. Caching allocators on CPU and GPU keep this fast.
- **Small on disk** — Quantization can make a model 4× smaller on disk, with little loss in accuracy.
- **Easy to integrate** — It has few dependencies and offers simple [Python](https://opennmt.net/CTranslate2/python/overview.html) and C++ APIs that cover most needs.
- **Configurable, interactive decoding** — [Advanced decoding features](https://opennmt.net/CTranslate2/decoding.html) let you autocomplete a partial sequence or return alternatives at a chosen spot.
- **Tensor parallelism for distributed inference** — Very large models can be split across multiple GPUs. See [this guide](docs/parallel.md#model-and-tensor-parallelism) to set it up.

Many of these features are hard to get from standard deep learning frameworks. That gap is why this project exists.

## Installation and usage

Install CTranslate2 with pip:

```bash
pip install ctranslate2
```

Use the Python module to convert models and to translate or generate text in a few lines:

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

See the [documentation](https://opennmt.net/CTranslate2) for more details and examples.

Have an AMD ROCm GPU? We provide separate Python wheels on the [releases page](https://github.com/OpenNMT/CTranslate2/releases/).

## Web Server

[ctranslate2-web-server](https://github.com/jordimas/ctranslate2-web-server) wraps CTranslate2 in a web server with an OpenAI-compatible REST API. If your app already talks to the OpenAI API, it can use CTranslate2 models with little change.

## Benchmarks

We translate the En->De test set _newstest2014_ with several models:

- [OpenNMT-tf WMT14](https://opennmt.net/Models-tf/#translation): a base Transformer trained with OpenNMT-tf on the WMT14 dataset (4.5M lines)
- [OpenNMT-py WMT14](https://opennmt.net/Models-py/#translation): a base Transformer trained with OpenNMT-py on the WMT14 dataset (4.5M lines)
- [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-de#opus-2020-02-26zip): a base Transformer trained with Marian on all OPUS data available on 2020-02-26 (81.9M lines)

The benchmark reports target tokens generated per second (higher is better), averaged over several runs. See the [benchmark scripts](tools/benchmark) for details and to reproduce the numbers.

**These numbers only hold for the exact setup used here. Both absolute and relative speed can change with different settings.**

#### CPU

|                                            | Tokens per second | Max. memory | BLEU  |
| ------------------------------------------ | ----------------- | ----------- | ----- |
| **OpenNMT-tf WMT14 model**                 |                   |             |       |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 209.2             | 2653MB      | 26.93 |
| **OpenNMT-py WMT14 model**                 |                   |             |       |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1)     | 275.8             | 2012MB      | 26.77 |
| - int8                                     | 323.3             | 1359MB      | 26.72 |
| CTranslate2 3.6.0                          | 658.8             | 849MB       | 26.77 |
| - int16                                    | 733.0             | 672MB       | 26.82 |
| - int8                                     | 860.2             | 529MB       | 26.78 |
| - int8 + vmap                              | 1126.2            | 598MB       | 26.64 |
| **OPUS-MT model**                          |                   |             |       |
| Transformers 4.26.1 (with PyTorch 1.13.1)  | 147.3             | 2332MB      | 27.90 |
| Marian 1.11.0                              | 344.5             | 7605MB      | 27.93 |
| - int16                                    | 330.2             | 5901MB      | 27.65 |
| - int8                                     | 355.8             | 4763MB      | 27.27 |
| CTranslate2 3.6.0                          | 525.0             | 721MB       | 27.92 |
| - int16                                    | 596.1             | 660MB       | 27.53 |
| - int8                                     | 696.1             | 516MB       | 27.65 |

Executed with 4 threads on a [_c5.2xlarge_](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

|                                            | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU  |
| ------------------------------------------ | ----------------- | --------------- | --------------- | ----- |
| **OpenNMT-tf WMT14 model**                 |                   |                 |                 |       |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 1483.5            | 3031MB          | 3122MB          | 26.94 |
| **OpenNMT-py WMT14 model**                 |                   |                 |                 |       |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1)     | 1795.2            | 2973MB          | 3099MB          | 26.77 |
| FasterTransformer 5.3                      | 6979.0            | 2402MB          | 1131MB          | 26.77 |
| - float16                                  | 8592.5            | 1360MB          | 1135MB          | 26.80 |
| CTranslate2 3.6.0                          | 6634.7            | 1261MB          | 953MB           | 26.77 |
| - int8                                     | 8567.2            | 1005MB          | 807MB           | 26.85 |
| - float16                                  | 10990.7           | 941MB           | 807MB           | 26.77 |
| - int8 + float16                           | 8725.4            | 813MB           | 800MB           | 26.83 |
| **OPUS-MT model**                          |                   |                 |                 |       |
| Transformers 4.26.1 (with PyTorch 1.13.1)  | 1022.9            | 4097MB          | 2109MB          | 27.90 |
| Marian 1.11.0                              | 3241.0            | 3381MB          | 2156MB          | 27.92 |
| - float16                                  | 3962.4            | 3239MB          | 1976MB          | 27.94 |
| CTranslate2 3.6.0                          | 5876.4            | 1197MB          | 754MB           | 27.92 |
| - int8                                     | 7521.9            | 1005MB          | 792MB           | 27.79 |
| - float16                                  | 9296.7            | 909MB           | 814MB           | 27.90 |
| - int8 + float16                           | 8362.7            | 813MB           | 766MB           | 27.90 |

Executed with CUDA 11 on a [_g5.xlarge_](https://aws.amazon.com/ec2/instance-types/g5/) Amazon EC2 instance equipped with a NVIDIA A10G GPU (driver version: 510.47.03).

#### Apple Metal — choosing a `compute_type`

On the Metal backend the fastest `compute_type` depends on the **model architecture**, not a single global default. The deciding factor is whether the model's hot ops have fused FP16 GPU kernels: if they do, FP16 activations stay on the GPU and win; if not, they pay an FP16↔FP32 conversion tax. Measured on an Apple M4 Max (full tables and methodology in [`METAL_BENCHMARKS.md`](METAL_BENCHMARKS.md)):

- **Decoder-only LLMs** (e.g. Qwen2.5) → **`float16`**, in _every_ regime. Its hot path (fused SDPA, RMSNorm, RoPE) is FP16-native, so FP16 wins both decode (up to ~3.6× vs the Accelerate CPU backend) and prefill (up to ~3.6×), and its lead grows with prompt length. `int8` is the _slowest_ Metal option here.
- **Encoder-decoder translation** (OPUS-MT, NLLB, M2M) → **`int8`** for decode / small batch, **`float32`** for large-batch prefill. These models route more ops through the CPU reference, so the FP16-activation types (`float16`, `int8_float16`) are a _pessimization_ — avoid them.

Rule of thumb: **LLM → `float16`; classic encoder-decoder → `int8` (decode) or `float32` (prefill).**

## Contributing

CTranslate2 is a community-driven project. We welcome contributions of all kinds:

- **New Model Support:** Help us implement more Transformer architectures.
- **Performance:** Propose optimizations for CPU or GPU kernels.
- **Bug Reports:** Open an issue if you find something not working as expected.
- **Documentation:** Improve our guides or add new examples.

Check out our [Contributing Guide](CONTRIBUTING.md) to learn how to set up your development environment.

## Additional resources

- [Documentation](https://opennmt.net/CTranslate2)
- [Forum](https://forum.opennmt.net)
- [Gitter](https://gitter.im/OpenNMT/CTranslate2)
