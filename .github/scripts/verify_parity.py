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
    content = content.replace('num_steps = 1000', 'num_steps = 1')
    content = content.replace("end='\\r'", "end='\\n'")
    content = content.replace('loss {loss.data:.4f}', 'loss {loss.data:.8f}')
    return content

def patch_ocaml(content):
    print("Patching OCaml port...")
    content = content.replace('gauss 0.0 std', '0.02')
    shuffle_pattern = re.compile(r'let docs_shuffled =.*?in Array.of_list \(shuffle d\)\n\s+in', re.DOTALL)
    content = shuffle_pattern.sub('let docs_shuffled = docs in', content)
    content = content.replace('let num_steps = 1000', 'let num_steps = 1')
    content = content.replace('%.4f', '%.8f')
    content = content.replace('\\r%!', '\\n%!')
    return content

def run_step(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stderr)
        return None
    
    # Extract loss from line like "step    1 /    1 | loss 3.29583687"
    for line in result.stdout.split('\n'):
        if "step    1 " in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return None

def main():
    py_content = download_reference()
    with open('ref.py', 'w') as f:
        f.write(patch_python(py_content))

    with open('microgpt.ml', 'r') as f:
        ml_content = f.read()
    with open('micro_test.ml', 'w') as f:
        f.write(patch_ocaml(ml_content))

    print("Running Python Reference...")
    py_loss = run_step("python3 ref.py")
    
    print("Running OCaml Port...")
    subprocess.run("ocamlopt -o micro_test_bin micro_test.ml", shell=True, check=True)
    ml_loss = run_step("./micro_test_bin")

    print(f"Python Step 1 Loss: {py_loss}")
    print(f"OCaml  Step 1 Loss: {ml_loss}")

    success = False
    if py_loss and ml_loss:
        if float(py_loss) == float(ml_loss):
            print("SUCCESS: Mathematical Parity Verified!")
            success = True
        else:
            print("FAILURE: Loss divergence detected!")
    else:
        print("FAILURE: Could not extract loss values.")

    # Cleanup
    for f in ['ref.py', 'micro_test.ml', 'micro_test_bin', 'micro_test.o', 'micro_test.cmx', 'micro_test.cmi']:
        if os.path.exists(f):
            os.remove(f)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
