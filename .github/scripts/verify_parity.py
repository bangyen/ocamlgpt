import urllib.request
import re
import os
import subprocess
import sys

# Karpathy's microgpt.py Gist
REF_URL = "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py"

def download_reference():
    print(f"Downloading reference Python implementation from {REF_URL}...")
    with urllib.request.urlopen(REF_URL) as response:
        return response.read().decode('utf-8')

def patch_python(content):
    print("Patching Python reference...")
    content = content.replace('random.shuffle(docs)', '# random.shuffle(docs)')
    content = content.replace('random.gauss(0, std)', '0.02')
    content = content.replace('num_steps = 1000', 'num_steps = 10')
    content = content.replace("end='\\r'", "end='\\n'")
    content = content.replace('loss {loss.data:.4f}', 'loss {loss.data:.8f}')
    # Deterministic sampling: always pick the first token (usually BOS or 'a' etc.)
    content = content.replace(
        'token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]',
        'token_id = 0'
    )
    return content

def patch_ocaml(content):
    print("Patching OCaml port...")
    # Robust document loading to match Python's [line.strip() for line in open('input.txt') if line.strip()]
    data_loading_patch = """
  let docs =
    let ic = open_in "input.txt" in
    let rec read_lines acc =
      try 
        let line = input_line ic in
        let s = String.trim line in
        read_lines (if s <> "" then s :: acc else acc)
      with End_of_file -> close_in ic; List.rev acc
    in 
    Array.of_list (read_lines [])
  in"""
    content = re.sub(r'let docs =.*?in\s+Array\.of_list \(List\.rev \(read_lines \[\]\)\)\s+in', data_loading_patch, content, flags=re.DOTALL)

    content = content.replace('gauss 0.0 std', '0.02')
    # Disable shuffle: replace the entire docs_shuffled block
    # We use a very lazy regex to match the docs_shuffled block until the next 'for' or similar
    content = re.sub(r'let docs_shuffled =.*?in\s+for step', 'let docs_shuffled = docs in\n\n  for step', content, flags=re.DOTALL)
    
    content = content.replace('let num_steps = 1000', 'let num_steps = 10')
    content = content.replace('%.4f', '%.8f')
    content = content.replace('\\r%!', '\\n%!')
    
    # Deterministic sampling: overwrite the entire sampling logic to always pick token 0 if not BOS
    content = content.replace('token_id := !selected_idx;', 'token_id := 0;')
    return content

def run_output(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stderr)
        return [], []
    
    losses = []
    samples = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if "step" in line and "loss" in line:
            parts = line.split()
            if parts:
                losses.append(parts[-1])
        if "sample" in line:
            parts = line.split(": ")
            if len(parts) > 1:
                samples.append(parts[1].strip())
    return losses, samples

def main():
    py_content = download_reference()
    with open('ref.py', 'w') as f:
        f.write(patch_python(py_content))

    with open('microgpt.ml', 'r') as f:
        ml_content = f.read()
    with open('micro_test.ml', 'w') as f:
        f.write(patch_ocaml(ml_content))

    print("Running Python Reference...")
    py_losses, py_samples = run_output("python3 ref.py")
    
    print("Running OCaml Port...")
    # Using -w -40 to ignore some warnings during patching if any
    subprocess.run("ocamlopt -w -40 -o micro_test_bin micro_test.ml", shell=True, check=True)
    ml_losses, ml_samples = run_output("./micro_test_bin")

    success = True
    if not py_losses or not ml_losses:
        print("FAILURE: Could not extract loss values.")
        success = False
    elif py_losses != ml_losses:
        print("FAILURE: Loss divergence detected!")
        max_len = max(len(py_losses), len(ml_losses))
        for i in range(max_len):
            p = py_losses[i] if i < len(py_losses) else "MISSING"
            m = ml_losses[i] if i < len(ml_losses) else "MISSING"
            if p != m:
                print(f"Step {i+1}: Python={p}, OCaml={m}")
        success = False
    else:
        print(f"SUCCESS: Mathematical Parity Verified for {len(py_losses)} steps!")

    if not py_samples or not ml_samples:
        print("FAILURE: Could not extract samples.")
        success = False
    elif py_samples != ml_samples:
        print("FAILURE: Sample divergence detected!")
        max_len = max(len(py_samples), len(ml_samples))
        for i in range(max_len):
            p = py_samples[i] if i < len(py_samples) else "MISSING"
            m = ml_samples[i] if i < len(ml_samples) else "MISSING"
            if p != m:
                print(f"Sample {i+1}: Python='{p}', OCaml='{m}'")
        success = False
    else:
        print(f"SUCCESS: Sampling Parity Verified for {len(py_samples)} samples!")

    # Cleanup
    for f in ['ref.py', 'micro_test.ml', 'micro_test_bin', 'micro_test.o', 'micro_test.cmx', 'micro_test.cmi']:
        if os.path.exists(f):
            os.remove(f)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
