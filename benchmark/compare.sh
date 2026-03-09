#!/bin/bash
echo "=== Python microgpt.py Step 1 ==="
# Modify benchmark/microgpt.py to run only 1 step
sed -i '' 's/num_steps = 1000/num_steps = 1/' benchmark/microgpt.py
python3 benchmark/microgpt.py

echo -e "\n=== OCaml ocamlgpt Step 1 ==="
# Run OCaml version (assuming it was already built with 1 step or we just read the first line)
dune exec ocamlgpt | head -n 4
