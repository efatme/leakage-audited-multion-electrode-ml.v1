#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

FILE_MANIFEST = "GITHUB_FILE_MANIFEST.csv"
SHA_MANIFEST = "GITHUB_SHA256SUMS.txt"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_tracked_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(
        p.decode("utf-8", errors="surrogateescape")
        for p in proc.stdout.split(b"\0")
        if p
    )

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild GITHUB_FILE_MANIFEST.csv and GITHUB_SHA256SUMS.txt."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    if args.repo_root is not None:
        repo_root = args.repo_root.expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        repo_root = here.parent.parent if here.parent.name == "software" else Path.cwd().resolve()

    if not (repo_root / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {repo_root}")

    tracked = git_tracked_files(repo_root)
    missing = [p for p in tracked if not (repo_root / p).is_file()]
    if missing:
        raise SystemExit(
            "Tracked files are missing from the working tree:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    csv_paths = [p for p in tracked if p not in {FILE_MANIFEST, SHA_MANIFEST}]
    csv_path = repo_root / FILE_MANIFEST
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "size_bytes", "sha256"])
        for rel in csv_paths:
            path = repo_root / rel
            writer.writerow([rel.replace("\\", "/"), path.stat().st_size, sha256_file(path)])

    sha_paths = [p for p in tracked if p != SHA_MANIFEST]
    if FILE_MANIFEST not in sha_paths:
        sha_paths.append(FILE_MANIFEST)
        sha_paths.sort()

    sha_path = repo_root / SHA_MANIFEST
    with sha_path.open("w", encoding="utf-8", newline="\n") as f:
        for rel in sha_paths:
            path = repo_root / rel
            f.write(f"{sha256_file(path)}  {rel.replace(chr(92), '/')}\n")

    print(f"Repository root : {repo_root}")
    print(f"CSV entries     : {len(csv_paths)}")
    print(f"SHA entries     : {len(sha_paths)}")
    print(f"Wrote           : {FILE_MANIFEST}")
    print(f"Wrote           : {SHA_MANIFEST}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
