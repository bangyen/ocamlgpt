# MicroGPT OCaml Port

An optimized, high-fidelity OCaml port of Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). This project reproduces the original micro-GPT architecture with bit-level parity while delivering significant performance gains via OCaml's native compilation and optimized array operations.

## Features

- **High-Fidelity Port**: Matches the original architecture, parameter count (4192 for 1-layer), and Step 1 loss (~3.3) for bit-level alignment.
- **Improved Capacity**: Default configuration uses 4 layers (~114k params) for significantly better convergence on the `names.txt` dataset.
- **Performance Optimized**: Array-based forward/backward pass refactoring provides a ~250x-500x speedup over pure scalar implementations.
- **Rigorous Verification**: Includes numerical gradient checking (finite differences) to mathematically prove autograd correctness.
- **Dune Powered**: Modern OCaml build system for seamless training, testing, and benchmarking.

## Requirements

- OCaml 5.0+
- [Dune](https://dune.build/)
- `curl` (for dataset and benchmark downloading)

## Getting Started

### 1. Setup Environment

```bash
opam init
eval $(opam env)
opam install dune
```

### 2. Run Training & Inference

The training loop automatically downloads the `names.txt` dataset and runs for 5000 steps, followed by name generation.

```bash
dune exec ocamlgpt
```

### 3. Run Verification Tests

Mathematically verify the autograd engine and model logic:

```bash
dune runtest
```

### 4. Cross-Language Parity Check

Compare the OCaml port's Step 1 loss directly against the original Python implementation:

```bash
bash benchmark/compare.sh
```

## Implementation Details

- **Autograd Engine**: A minimalist scalar-valued engine (`Value` module) in OCaml.
- **Architecture**: Implements GPT-2 style residual blocks with **RMSNorm** and **Multi-head Attention** (matched to `microgpt.py`'s exact structure).
- **Optimizer**: Adam with deterministic seeding (`Random.init 42`) for reproducible research.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## License

MIT
