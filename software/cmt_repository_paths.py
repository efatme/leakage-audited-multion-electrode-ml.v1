# Clean-room repository path helper.

from __future__ import annotations

import os
from pathlib import Path


SECTION_MAP: dict[str, dict[str, str]] = {
    "01": {
        "audit": "results/audits/notebook_01",
        "logs": "provenance/notebook_01",
        "metadata": "provenance/notebook_01",
        "processed": "data/processed/notebook_01",
        "raw": "data/raw/notebook_01"
    },
    "02": {
        "audit": "results/audits/notebook_02",
        "logs": "provenance/notebook_02",
        "metadata": "provenance/notebook_02",
        "processed": "data/processed/notebook_02"
    },
    "03": {
        "audit": "results/audits/notebook_03",
        "logs": "provenance/notebook_03",
        "metadata": "provenance/notebook_03",
        "processed": "data/processed/notebook_03",
        "software": "software/notebook_03",
        "tests": "software/tests/notebook_03"
    },
    "04": {
        "audit": "results/audits/notebook_04",
        "logs": "provenance/notebook_04",
        "metadata": "provenance/notebook_04",
        "processed": "data/processed/notebook_04"
    },
    "05": {
        "audit": "results/audits/notebook_05",
        "checkpoints": ".runtime_cache/notebook_05/checkpoints",
        "logs": "provenance/notebook_05",
        "metadata": "provenance/notebook_05",
        "metrics": "results/metrics/notebook_05",
        "models": "results/models/notebook_05",
        "processed": "data/processed/notebook_05"
    },
    "06": {
        "audit": "results/audits/notebook_06",
        "logs": "provenance/notebook_06",
        "metadata": "provenance/notebook_06",
        "metrics": "results/metrics/notebook_06",
        "processed": "data/processed/notebook_06"
    },
    "07": {
        "audit": "results/audits/notebook_07",
        "checkpoints": ".runtime_cache/notebook_07/checkpoints",
        "logs": "provenance/notebook_07",
        "metadata": "provenance/notebook_07",
        "processed": "data/processed/notebook_07"
    },
    "08": {
        "processed": "data/processed/notebook_08"
    },
    "09": {
        "audit": "results/audits/notebook_09",
        "logs": "provenance/notebook_09",
        "manual_inputs": "provenance/supplementary/notebook_09/manual_inputs",
        "metadata": "provenance/notebook_09",
        "processed": "data/processed/notebook_09"
    },
    "11": {
        "audit": "results/audits/notebook_11",
        "dft_inputs": "documentation/notebook_11/dft_inputs",
        "exact_structures": "provenance/supplementary/notebook_11/exact_structures",
        "logs": "provenance/notebook_11",
        "metadata": "provenance/notebook_11",
        "processed": "data/processed/notebook_11"
    },
    "12": {
        "audit": "results/audits/notebook_12",
        "logs": "provenance/notebook_12",
        "metadata": "provenance/notebook_12",
        "pilot_dft_package": "documentation/notebook_12/pilot_dft_package",
        "processed": "data/processed/notebook_12"
    }
}


def find_repository_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for root in [candidate, *candidate.parents]:
        if (
            (root / "notebooks").is_dir()
            and (root / "data").is_dir()
            and (root / "results").is_dir()
            and (root / "provenance").is_dir()
        ):
            return root
    raise FileNotFoundError("Could not locate the clean-room repository root.")


class ArtifactNamespace:
    def __init__(self, notebook_number: str, repository_root: Path | None = None) -> None:
        self.notebook_number = str(notebook_number).zfill(2)
        self.repository_root = repository_root or find_repository_root()
        if self.notebook_number not in SECTION_MAP:
            raise KeyError(f"No repository section map for notebook {self.notebook_number}.")

    def __truediv__(self, relative: str | Path) -> Path:
        relative_path = Path(relative)
        if not relative_path.parts:
            raise ValueError("Empty artifact-relative path.")
        section = relative_path.parts[0]
        notebook_map = SECTION_MAP[self.notebook_number]
        if section not in notebook_map:
            raise KeyError(
                f"Section {section!r} is not mapped for notebook {self.notebook_number}."
            )
        base = self.repository_root / notebook_map[section]
        return base.joinpath(*relative_path.parts[1:])

    def exists(self) -> bool:
        return any(
            (self.repository_root / relative).exists()
            for relative in SECTION_MAP[self.notebook_number].values()
            if not relative.startswith(".runtime_cache/")
        )

    def is_dir(self) -> bool:
        return self.exists()

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        for relative in SECTION_MAP[self.notebook_number].values():
            (self.repository_root / relative).mkdir(parents=parents, exist_ok=exist_ok)

    def resolve(self) -> Path:
        return self.repository_root

    def __str__(self) -> str:
        return f"ArtifactNamespace(notebook_{self.notebook_number})"

    def __fspath__(self) -> str:
        return os.fspath(self.repository_root)


def artifact_namespace(notebook_number: str, repository_root: Path | None = None) -> ArtifactNamespace:
    return ArtifactNamespace(notebook_number, repository_root)


def runtime_cache_root(repository_root: Path | None = None) -> Path:
    root = repository_root or find_repository_root()
    configured = os.environ.get("CMT_RUNTIME_CACHE_DIR")
    path = Path(configured).expanduser() if configured else root / ".runtime_cache"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
