from pathlib import Path
import csv
import hashlib
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'papers' / 'cmt' / 'publication' / 'outputs'
MANIFEST = OUT / 'publication_outputs_sha256.csv'

required_fig_stems = [
    'Fig1_information_provenance_and_eligibility',
    'Fig2_stage_dependent_learnability',
    'Fig3_incremental_information_value',
    'Fig4_chemical_domain_robustness',
    'Fig5_applicability_domain_diagnostic',
    'Fig6_sodium_triage_dft_handoff',
]
required_tables = [
    'Table1_target_level_summary.csv',
    'TableS1_stage_c_full_skill_bootstrap.csv',
    'TableS2_threshold_sensitivity.csv',
    'TableS3_domain_robustness_summary.csv',
    'TableS4_estimator_sensitivity.csv',
]
required_sources = [
    'source_data/Fig5_applicability_domain_source.csv',
    'source_data/Fig6_sodium_candidates_source.csv',
    'source_data/Fig6_dft_voltage_source.csv',
]
required_text = [
    'figure_captions_overlap_controlled.md',
    'TEXT_OVERLAP_FIREWALL.md',
    'caption_exact_overlap_audit.csv',
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

failures = []
if not OUT.exists():
    failures.append(f'missing output directory: {OUT}')
if not MANIFEST.exists():
    failures.append(f'missing checksum manifest: {MANIFEST}')

manifest = {}
if MANIFEST.exists():
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            manifest[row['file']] = (row['sha256'], int(row['bytes']))

required_files = []
for stem in required_fig_stems:
    required_files += [f'{stem}.pdf', f'{stem}.svg', f'{stem}.tif', f'{stem}_preview.png']
required_files += required_tables + required_text + required_sources + ['Fig2_plot_data.csv']

for name in required_files:
    p = OUT / name
    if not p.exists():
        failures.append(f'missing: {name}')
        continue
    if p.stat().st_size == 0:
        failures.append(f'empty file: {name}')
    if name not in manifest:
        failures.append(f'not listed in checksum manifest: {name}')
        continue
    expected_hash, expected_bytes = manifest[name]
    actual_hash = sha256(p)
    actual_bytes = p.stat().st_size
    if actual_hash != expected_hash:
        failures.append(f'checksum mismatch: {name}')
    if actual_bytes != expected_bytes:
        failures.append(f'byte-size mismatch: {name}')

# Minimal SVG structural checks. These do not use Matplotlib or any font library.
for stem in required_fig_stems:
    p = OUT / f'{stem}.svg'
    if not p.exists():
        continue
    try:
        root = ET.parse(p).getroot()
        tag = root.tag.split('}')[-1]
        if tag != 'svg':
            failures.append(f'not an SVG root element: {p.name}')
        if 'viewBox' not in root.attrib:
            failures.append(f'missing SVG viewBox: {p.name}')
    except Exception as e:
        failures.append(f'cannot parse SVG {p.name}: {e}')

# Caption-overlap audit must contain no flagged exact 8-word matches.
audit = OUT / 'caption_exact_overlap_audit.csv'
if audit.exists():
    with audit.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # The controlled audit file is expected to be header-only when there are no matches.
    if rows:
        failures.append(f'caption overlap audit contains {len(rows)} flagged match row(s)')

if failures:
    print('Publication verification: FAIL')
    for x in failures:
        print(' -', x)
    sys.exit(1)

print('Publication verification: PASS')
print('Figures verified: 6 x PDF/SVG/TIFF/PNG-preview')
print('Tables verified: 1 main + 4 SI')
print('Caption exact-overlap audit: 0 flagged rows')
print('Matplotlib/font-manager not used')
print('Output:', OUT)
