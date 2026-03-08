# MicroGPT OCaml Port

An OCaml port of Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). This is a minimalist, dependency-free implementation of a GPT model, including a scalar-based autograd engine, all in a single OCaml file.

## Features

- **Autograd Engine**: A minimalist scalar-valued autograd engine (`Value` module) with support for basic operations (`+`, `-`, `*`, `/`, `pow`, `log`, `exp`, `relu`) and backpropagation via topological sort.
- **GPT Architecture**: Faithfully reproduces the micro-GPT architecture:
  - RMSNorm (no biases)
  - Multi-head Attention
  - MLP with ReLU activation
  - Linear layers for embeddings and heads
- **Adam Optimizer**: Complete with first and second moment buffers and linear learning rate decay.
- **Tokenizer**: Simple character-level tokenizer.
- **Single File**: The entire implementation is contained in `microgpt.ml`.

## Requirements

- OCaml 5.0+ (Tested on 5.4.0)
- `opam` (optional, for environment management)

## Getting Started

### 1. Setup Environment

If you haven't already initialized your OCaml environment, you can do so with `opam`:

```bash
opam init
eval $(opam env)
```

### 2. Run Training

You can run the script directly using the OCaml interpreter:

```bash
ocaml microgpt.ml
```

By default, the script looks for `input.txt`. If not found, it uses a small fallback dataset.

### 3. Compile for Performance

For significantly faster training, compile the script to a native binary:

```bash
ocamlopt -o microgpt microgpt.ml
./microgpt
```

## Implementation Details

The OCaml port closely follows the structure of the original Python script while adhering to OCaml's functional paradigms. Key differences include:
- Use of recursive functions for loop control (e.g., in the inference loop to handle early exit).
- Explicit type annotations and record field handling for the `Value` module.
- `Hashtbl` for storing layer parameters to maintain a structure similar to Python's `state_dict`.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
