from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / 'papers' / 'cmt' / 'publication'
OUT = PUB / 'outputs'
ART = PUB / 'submission_artwork'
ART_MANIFEST = ART / 'submission_artwork_sha256.csv'
RUN_MANIFEST = ART / 'submission_artwork_run_manifest.json'

FIGURE_STEMS = [
    'Fig1_information_provenance_and_eligibility_submission',
    'Fig2_stage_dependent_learnability_submission',
    'Fig3_incremental_information_value_submission',
    'Fig4_chemical_domain_robustness_submission',
    'Fig5_applicability_domain_diagnostic_submission',
    'Fig6_sodium_triage_dft_handoff_submission',
]

PUBLICATION_TABLES = [
    'Fig2_plot_data.csv',
    'Table1_target_level_summary.csv',
    'TableS1_stage_c_full_skill_bootstrap.csv',
    'TableS2_threshold_sensitivity.csv',
    'TableS3_domain_robustness_summary.csv',
    'TableS4_estimator_sensitivity.csv',
]

SOURCE_TABLES = [
    'source_data/Fig5_applicability_domain_source.csv',
    'source_data/Fig6_sodium_candidates_source.csv',
    'source_data/Fig6_dft_voltage_source.csv',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_nonempty(path: Path, label: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f'missing {label}: {path.relative_to(ROOT)}')
        return
    if not path.is_file():
        failures.append(f'not a file ({label}): {path.relative_to(ROOT)}')
        return
    if path.stat().st_size == 0:
        failures.append(f'empty {label}: {path.relative_to(ROOT)}')


def read_artwork_manifest(failures: list[str]) -> dict[str, tuple[str, int]]:
    manifest: dict[str, tuple[str, int]] = {}
    if not ART_MANIFEST.exists():
        failures.append(f'missing artwork checksum manifest: {ART_MANIFEST.relative_to(ROOT)}')
        return manifest
    try:
        with ART_MANIFEST.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                manifest[row['file']] = (row['sha256'], int(row['bytes']))
    except Exception as exc:
        failures.append(f'cannot read artwork checksum manifest: {exc}')
    return manifest


def verify_artwork(failures: list[str]) -> None:
    manifest = read_artwork_manifest(failures)
    expected = []
    for stem in FIGURE_STEMS:
        expected.extend([
            f'{stem}.pdf', f'{stem}.svg', f'{stem}.tif', f'{stem}_preview.png'
        ])

    for name in expected:
        path = ART / name
        require_nonempty(path, 'artwork file', failures)
        if not path.exists() or path.stat().st_size == 0:
            continue
        if name not in manifest:
            failures.append(f'not listed in artwork checksum manifest: {name}')
            continue
        expected_hash, expected_bytes = manifest[name]
        if sha256(path) != expected_hash:
            failures.append(f'artwork checksum mismatch: {name}')
        if path.stat().st_size != expected_bytes:
            failures.append(f'artwork byte-size mismatch: {name}')

    for stem in FIGURE_STEMS:
        path = ART / f'{stem}.svg'
        if not path.exists():
            continue
        try:
            root = ET.parse(path).getroot()
            if root.tag.split('}')[-1] != 'svg':
                failures.append(f'not an SVG root element: {path.name}')
            if 'viewBox' not in root.attrib:
                failures.append(f'missing SVG viewBox: {path.name}')
        except Exception as exc:
            failures.append(f'cannot parse SVG {path.name}: {exc}')


def verify_run_manifest(failures: list[str]) -> None:
    require_nonempty(RUN_MANIFEST, 'artwork run manifest', failures)
    if not RUN_MANIFEST.exists() or RUN_MANIFEST.stat().st_size == 0:
        return
    try:
        run = json.loads(RUN_MANIFEST.read_text(encoding='utf-8'))
    except Exception as exc:
        failures.append(f'cannot parse artwork run manifest: {exc}')
        return

    source_files = run.get('source_files')
    if not isinstance(source_files, dict) or not source_files:
        failures.append('artwork run manifest has no source_files mapping')
        return

    for raw_name, expected_hash in source_files.items():
        rel = Path(*str(raw_name).replace('\\', '/').split('/'))
        path = ROOT / rel
        if not path.exists():
            failures.append(f'frozen artwork input missing: {raw_name}')
            continue
        if sha256(path) != expected_hash:
            failures.append(f'frozen artwork input checksum mismatch: {raw_name}')


def verify_publication_tables(failures: list[str]) -> None:
    for rel in PUBLICATION_TABLES + SOURCE_TABLES:
        path = OUT / rel
        require_nonempty(path, 'publication data file', failures)
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with path.open(newline='', encoding='utf-8-sig') as f:
                header = next(csv.reader(f), None)
            if not header:
                failures.append(f'CSV has no header: {path.relative_to(ROOT)}')
        except Exception as exc:
            failures.append(f'cannot read CSV {path.relative_to(ROOT)}: {exc}')


def main() -> int:
    failures: list[str] = []

    if not PUB.exists():
        failures.append(f'missing publication directory: {PUB}')
    if not OUT.exists():
        failures.append(f'missing publication data directory: {OUT}')
    if not ART.exists():
        failures.append(f'missing submission artwork directory: {ART}')

    require_nonempty(PUB / 'FIGURE_SOURCE_MAP.csv', 'figure source map', failures)
    verify_artwork(failures)
    verify_run_manifest(failures)
    verify_publication_tables(failures)

    if failures:
        print('CMT publication verification: FAIL')
        for item in failures:
            print(' -', item)
        return 1

    print('CMT publication verification: PASS')
    print('Final artwork verified: 6 x PDF/SVG/TIFF/PNG-preview')
    print('Artwork SHA256 manifest: verified')
    print('Frozen artwork inputs: verified against run manifest')
    print('Publication data: 1 main table + 4 SI tables + Figure 2 plot data')
    print('Supporting source data: Figure 5 + Figure 6 sodium/DFT tables')
    print('SVG structure: verified')
    print('Matplotlib/model fitting: not used')
    print(f'Publication directory: {PUB}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
