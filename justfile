# default task - list all available tasks
default:
    @just --list

# build both implementations
build:
    @ocamlopt -w +a-4-40-42-70 -o microgpt microgpt.ml
    @ocamlopt -o fastgpt fastgpt.ml

# build and run the optimized fastgpt
run-fast: build
    @./fastgpt

# build and run the minimalist microgpt
run-micro: build
    @./microgpt

# verify bit-level parity between microgpt and fastgpt
check: build
    @./microgpt | head -n 12 > micro_out.txt
    @./fastgpt | head -n 12 > fast_out.txt
    @if diff micro_out.txt fast_out.txt; then \
        echo "Bit-level parity verified!"; \
    else \
        echo "Bit-level parity verification failed!"; \
        exit 1; \
    fi
    @rm micro_out.txt fast_out.txt

# verify mathematical parity with carbon-reference Python implementation
verify:
    @python3 .github/scripts/verify_parity.py

# clean build artifacts
clean:
    @rm -f microgpt fastgpt *.o *.cmx *.cmi *.obj *.exe
    @rm -f micro_test* ref.py
