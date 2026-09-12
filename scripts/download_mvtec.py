"""Download the official bottle subset; does not redistribute dataset."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
URL="https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz"
def main():
    root=Path("data")
    archive=root/"downloads/bottle.tar.xz"
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temporary=archive.with_suffix(".part")
        urllib.request.urlretrieve(URL,temporary)
        temporary.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    expected="726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3"
    if digest!=expected:
        raise ValueError("Archive checksum mismatch; inspect download before extraction")
    target=root/"mvtec"
    if (target/"bottle").exists():
        print("bottle already exists; no overwrite")
        return
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,"r:xz") as t:
        t.extractall(target,filter="data")
    (target/"dataset.json").write_text(json.dumps({
        "source":"MVTec AD official bottle category","url":"https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads",
        "license":"CC BY-NC-SA 4.0","archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2),encoding="utf-8")
    print("Downloaded for non-commercial research. See MVTec license; do not commit data to GitHub.")
if __name__=="__main__":
    main()
