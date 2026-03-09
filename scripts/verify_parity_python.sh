#!/bin/bash
set -e

# Karpathy's microgpt.py Gist (Stable Version)
REF_URL="https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py"

echo "Downloading reference Python implementation..."
curl -sL $REF_URL > ref.py

# Check if file is valid
if grep -q "404: Not Found" ref.py; then
    echo "Error: Reference implementation not found."
    exit 1
fi

echo "Patching implementations..."
python3 - <<'EOF'
import re
import os

# Patch Python
with open('ref.py', 'r') as f:
    py = f.read()

# Remove shuffle
py = py.replace('random.shuffle(docs)', '# random.shuffle(docs)')
# Constant weights
py = py.replace('random.gauss(0, std)', '0.02')
# Single step
py = py.replace('num_steps = 1000', 'num_steps = 1')
# Fix carriage return for CI/grep
py = py.replace("end='\\r'", "end='\\n'")
# Increase precision
py = py.replace('loss {loss.data:.4f}', 'loss {loss.data:.8f}')

with open('ref.py', 'w') as f:
    f.write(py)

# Patch OCaml
with open('microgpt.ml', 'r') as f:
    ml = f.read()

# Replace matrix initialization
ml = ml.replace('gauss 0.0 std', '0.02')
# Replace shuffle block with identity using more robust regex
# This pattern matches the specific docs_shuffled block in microgpt.ml
shuffle_pattern = re.compile(r'let docs_shuffled =.*?in Array.of_list \(shuffle d\)\n\s+in', re.DOTALL)
ml = shuffle_pattern.sub('let docs_shuffled = docs in', ml)
# Run for 1 step
ml = ml.replace('let num_steps = 1000', 'let num_steps = 1')
# Increase precision
ml = ml.replace('%.4f', '%.8f')
# Use newline instead of \r for grep
ml = ml.replace('\\r%!', '\\n%!')

with open('micro_test.ml', 'w') as f:
    f.write(ml)
EOF

echo "Running Python Reference..."
# Note: we use grep to find the loss value exactly
PY_LOSS=$(python3 ref.py | grep "step    1 " | head -n 1 | awk '{print $NF}' | tr -d '\r\n ')

echo "Running OCaml Port..."
ocamlopt -o micro_test_bin micro_test.ml
OCAML_LOSS=$(./micro_test_bin | grep "step    1 " | awk '{print $NF}' | tr -d '\r\n ')

echo "Python Step 1 Loss: $PY_LOSS"
echo "OCaml  Step 1 Loss: $OCAML_LOSS"

# Verification
RES=1
if python3 -c "import sys; exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" "$PY_LOSS" "$OCAML_LOSS"; then
    echo "SUCCESS: Mathematical Parity Verified!"
    RES=0
else
    echo "FAILURE: Loss divergence detected!"
    RES=1
fi

# Clean up
rm ref.py micro_test.ml micro_test_bin micro_test.o micro_test.cmx micro_test.cmi

exit $RES
