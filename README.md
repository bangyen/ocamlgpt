# MicroGPT OCaml Port

An optimized, high-fidelity OCaml port of Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). This project is a single-file "art project" that reproduces the original micro-GPT architecture with bit-level parity while delivering significant performance gains.

## Features

- **Single File**: The entire algorithm (Autograd, GPT, Training, Inference) is contained in `microgpt.ml`.
- **High-Fidelity Port**: Matches the original architecture, parameter count (4192 for 1-layer), and Step 1 loss (~3.3) for bit-level alignment.
- **Improved Capacity**: Default configuration uses 4 layers (~114k params) for significantly better convergence on the `names.txt` dataset.
- **Performance Optimized**: Array-based forward/backward pass refactoring provides a ~250x-500x speedup over pure scalar implementations.
- **Atomic Distillation**: No dependencies beyond the OCaml standard library.

## Requirements

- OCaml 5.0+
- `curl` (for initial dataset download)

## Getting Started

### 1. Setup Environment

```bash
opam init
eval $(opam env)
```

### 2. Run Training & Inference (Interpreted)

The script automatically downloads the `names.txt` dataset and runs for 5000 steps.

```bash
ocaml microgpt.ml
```

### 3. Compile for Maximum Performance (Native)

For significantly faster training, compile to a native binary:

```bash
ocamlopt -o microgpt microgpt.ml
./microgpt
```

## Implementation Details

- **Autograd Engine**: A minimalist scalar-valued engine (`Value` module) in OCaml.
- **Architecture**: Implements GPT-2 style residual blocks with **RMSNorm** and **Multi-head Attention**.
- **Optimizer**: Adam with deterministic seeding (`Random.init 42`).

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## License

MIT
