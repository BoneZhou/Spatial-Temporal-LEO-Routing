"""Verify shipped data integrity and the source-only release boundary (stdlib)."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(ROOT / "data") or not path.is_file():
            raise RuntimeError("Missing or invalid data path: " + record["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError("Data hash mismatch: " + record["path"])
    expected = {record["path"] for record in manifest["files"]}
    actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "data").rglob("*")
              if path.is_file() and path.name not in ("README.md", "manifest.json")}
    if actual != expected:
        raise RuntimeError("Data manifest coverage mismatch")
    blocked = {".pt", ".pth", ".pyc", ".pyo", ".log", ".tmp", ".partial"}
    leaks = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or relative.parts[0] in ("outputs", ".venv", "__pycache__"):
            continue
        if "__pycache__" in relative.parts:
            continue
        if path.suffix in blocked:
            leaks.append(str(relative))
        if path.suffix in (".py", ".md", ".json", ".csv", ".txt"):
            text = path.read_text(encoding="utf-8-sig")
            patterns = (r"[A-Za-z]:[\\/]+Users[\\/]+",
                        r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
                        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                        r"\bAKIA[A-Z0-9]{16}\b")
            if any(re.search(pattern, text) for pattern in patterns):
                leaks.append(str(relative))
    if leaks:
        raise RuntimeError("Review these release files: " + ", ".join(sorted(set(leaks))))
    print(f"Verified {len(expected)} reference files; source/data privacy scan passed.")


if __name__ == "__main__":
    main()
