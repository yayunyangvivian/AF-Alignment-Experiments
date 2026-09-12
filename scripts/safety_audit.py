"""Audit only the explicit public-file inventory; never stage files automatically."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','outputs','__pycache__','plot_cache'}
ALLOWED={'.py','.csv','.json','.txt','.rst'}
FORBIDDEN={'.tex','.md','.pdf','.pt','.pth','.npy','.npz','.pkl','.joblib','.mat','.hea','.dat','.avi','.mp4','.log','.aux'}
def main():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts)&EXCLUDE))
    findings=[]
    secrets=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[0-9A-Z]{16}',r'sk-[A-Za-z0-9]{24,}',r'(?i)(?:api_key|password|access_token|secret_key)\s*[:=]\s*[\x22\x27][^\x22\x27]{8,}[\x22\x27]']
    absolute=re.compile(r'/(?:Users|home|private/var|Volumes|mnt)/|[A-Za-z]:\\Users\\|~/')
    for p in files:
        rel=str(p.relative_to(ROOT))
        if p.is_symlink():findings.append([rel,'symlink'])
        if p.suffix.lower() in FORBIDDEN or (p.suffix not in ALLOWED and p.name!='.gitignore'):findings.append([rel,'excluded type'])
        if p.stat().st_size>2_000_000:findings.append([rel,'larger than 2 MB'])
        try:text=p.read_text()
        except UnicodeDecodeError:
            findings.append([rel,'unexpected binary file']);continue
        if p.name!='safety_audit.py':
            for i,line in enumerate(text.splitlines(),1):
                if absolute.search(line):findings.append([rel,i,'absolute local path'])
                if any(re.search(s,line) for s in secrets):findings.append([rel,i,'possible secret; value suppressed'])
        if p.suffix=='.csv':
            header=text.splitlines()[0].split(',')
            if any(c in header for c in ['record_id','patient_id','video_filename','test_label','age','sex','diagnosis_codes']):findings.append([rel,'possible record-level data'])
    inventory=[str(p.relative_to(ROOT)) for p in files]
    out=ROOT/'outputs/safety';out.mkdir(parents=True,exist_ok=True)
    (out/'allowlist.txt').write_text('\n'.join(inventory)+'\n')
    report=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),tex=sum(p.suffix=='.tex' for p in files),markdown=sum(p.suffix=='.md' for p in files),pdf=sum(p.suffix=='.pdf' for p in files),findings=findings,inventory=inventory)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'},indent=2))
    if findings:sys.exit(1)
if __name__=='__main__':main()
