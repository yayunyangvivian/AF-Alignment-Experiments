from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check_inputs(ptb,echo):
    expected=json.loads((ROOT/'configs/retained_input_hashes.json').read_text())
    actual={}
    for logical,h in expected.items():
        kind,name=logical.split('/')
        p=(ptb if kind=='ptb' else echo)/name
        if not p.is_file():raise FileNotFoundError(f'Required input: {logical}')
        actual[logical]=sha(p)
        if actual[logical]!=h:raise ValueError(f'Input hash mismatch: {logical}; do not silently use another configuration')
    return actual
def output_dir(path):
    p=Path(path).resolve()
    if p.exists() and any(p.iterdir()):raise FileExistsError('Use a new empty output directory')
    p.mkdir(parents=True,exist_ok=True)
    return p
