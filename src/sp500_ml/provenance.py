import hashlib
from pathlib import Path

def compatible_source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compatible_source_identity(path):
    return Path(path).name
