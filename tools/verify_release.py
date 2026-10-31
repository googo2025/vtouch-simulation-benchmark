"""Verify the frozen release bytes. Does not run any scientific model."""
from pathlib import Path
import hashlib,json,gzip,struct
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance/release-files.json').read_text())
errors=[]
for row in manifest['files']:
 p=root/row['path']
 if not p.is_file():errors.append(row['path']+': missing');continue
 b=p.read_bytes()
 if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:errors.append(row['path']+': mismatch')
 if p.name.endswith(('.bin.gz','.f32.gz')):
  try:gzip.decompress(b)
  except Exception:errors.append(row['path']+': gzip unreadable')
if errors:raise SystemExit('\n'.join(errors))
print('PASS: '+str(len(manifest['files']))+' frozen files, exact bytes and gzip readability.')
