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
    content = re.sub(r'random\.shuffle\(docs\)', '# random.shuffle(docs)', content)
    content = re.sub(r'random\.gauss\([^)]+\)', '0.02', content)
    content = re.sub(r'\bnum_steps\s*=\s*\d+', 'num_steps = 10', content)
    content = content.replace("end='\\r'", "end='\\n'")
    content = re.sub(r'loss\s*\{\s*loss\.data\s*:\s*\.\d+f\s*\}', 'loss {loss.data:.8f}', content)
    # Deterministic sampling: always pick the first token
    content = re.sub(r'token_id\s*=\s*random\.choices.*', 'token_id = 0', content)
    return content

def patch_ocaml(content):
    print("Patching OCaml port...")
    content = re.sub(r'gauss\s+0\.0\s+std', '0.02', content)
    content = re.sub(r'let gauss.*?\(2\.0 \*\. Float\.pi \*\. u2\)', 'let gauss mean std = 0.02', content, flags=re.DOTALL)
    
    # Disable shuffle by turning it into identity
    content = re.sub(r'let a = Array\.copy docs in.*?a\s+in', 'let a = docs in\n    a\n  in', content, flags=re.DOTALL)
    
    content = re.sub(r'let num_steps\s*=\s*\d+', 'let num_steps = 10', content)
    content = content.replace('%.4f', '%.8f')
    content = content.replace('\\r%!', '\\n%!')
    
    # Deterministic sampling
    content = re.sub(r'token_id\s*:=\s*sample_prob\s+0\s+0\.0', 'token_id := 0', content)
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
