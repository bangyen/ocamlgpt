# Ensure benchmark folder and microgpt.py exist
mkdir -p benchmark
if [ ! -f benchmark/microgpt.py ]; then
    echo "Downloading original microgpt.py..."
    curl -L https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py -o benchmark/microgpt.py
    # Patch Python script for a quick 1-step comparison
    sed -i '' 's/num_steps = 1000/num_steps = 1/' benchmark/microgpt.py
fi

echo "=== Python microgpt.py Step 1 ==="
python3 benchmark/microgpt.py | head -n 4

echo -e "\n=== OCaml ocamlgpt Step 1 ==="
# Run OCaml version (assuming it was already built with 1 step or we just read the first line)
dune exec ocamlgpt | head -n 4
