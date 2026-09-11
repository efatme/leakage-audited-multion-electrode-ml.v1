from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

# Headless backend is essential on Windows Server / remote Anaconda sessions.
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib as mpl
mpl.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve()

parser = argparse.ArgumentParser(
    description='Generate CMT submission figures from the frozen publication source data.'
)
parser.add_argument(
    '--repo-root', type=Path, default=None,
    help='Repository root. Recommended on Windows Server to avoid working-directory ambiguity.'
)
parser.add_argument(
    '--out-dir', type=Path, default=None,
    help='Optional output directory. Default: papers/cmt/publication/submission_artwork.'
)
parser.add_argument(
    '--check-only', action='store_true',
    help='Validate Python environment and required frozen inputs, then exit without plotting.'
)
args = parser.parse_args()


def _looks_like_repo_root(p: Path) -> bool:
    return (p / 'papers' / 'cmt' / 'results' / 'stage_f' / 'stage_f_master_result_matrix.csv').exists()


def _resolve_repo_root() -> Path:
    if args.repo_root is not None:
        root = args.repo_root.expanduser().resolve()
        if not _looks_like_repo_root(root):
            raise FileNotFoundError(
                f'--repo-root does not look like the CMT repository root: {root}\n'
                'Expected: papers/cmt/results/stage_f/stage_f_master_result_matrix.csv'
            )
        return root

    # Search upward from both the script location and current working directory.
    candidates = []
    for start in (HERE.parent, Path.cwd().resolve()):
        candidates.append(start)
        candidates.extend(start.parents)
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if _looks_like_repo_root(p):
            return p

    raise FileNotFoundError(
        'Could not locate the CMT repository root automatically.\n'
        f'Script: {HERE}\nCurrent directory: {Path.cwd().resolve()}\n'
        'Run again with --repo-root "C:\\path\\to\\repository".'
    )


ROOT = _resolve_repo_root()
RES = ROOT / 'papers' / 'cmt' / 'results'
FROZEN_OUT = ROOT / 'papers' / 'cmt' / 'publication' / 'outputs'
OUT = (args.out_dir.expanduser().resolve() if args.out_dir else
       ROOT / 'papers' / 'cmt' / 'publication' / 'submission_artwork')
OUT.mkdir(parents=True, exist_ok=True)

MASTER = RES / 'stage_f' / 'stage_f_master_result_matrix.csv'
DOMAIN = RES / 'stage_d' / 'stage_d_domain_summary.csv'
FIG5_SRC = FROZEN_OUT / 'source_data' / 'Fig5_applicability_domain_source.csv'
FIG6_CAND = FROZEN_OUT / 'source_data' / 'Fig6_sodium_candidates_source.csv'
FIG6_DFT = FROZEN_OUT / 'source_data' / 'Fig6_dft_voltage_source.csv'

required = [MASTER, DOMAIN, FIG5_SRC, FIG6_CAND, FIG6_DFT]
missing = [p for p in required if not p.exists()]

print('CMT submission figure renderer')
print(f'Python       : {sys.version.split()[0]}')
print(f'Executable   : {sys.executable}')
print(f'Platform     : {platform.platform()}')
print(f'Matplotlib   : {mpl.__version__} (backend={mpl.get_backend()})')
print(f'NumPy        : {np.__version__}')
print(f'Pandas       : {pd.__version__}')
print(f'Repository   : {ROOT}')
print(f'Output       : {OUT}')
for p in required:
    print(f'[{"OK" if p.exists() else "MISSING"}] {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}')

if missing:
    raise FileNotFoundError('Missing required frozen source files:\n' + '\n'.join(map(str, missing)))

if args.check_only:
    print('CHECK PASSED: environment and all required frozen inputs are available.')
    raise SystemExit(0)

m = pd.read_csv(MASTER)
d = pd.read_csv(DOMAIN)
ad = pd.read_csv(FIG5_SRC)
cand = pd.read_csv(FIG6_CAND)
dft = pd.read_csv(FIG6_DFT)

order = [
    'average_voltage', 'capacity_grav', 'capacity_vol', 'energy_grav', 'energy_vol',
    'max_delta_volume', 'stability_charge', 'stability_discharge', 'stability_worst'
]
labels = {
    'average_voltage': 'Average voltage',
    'capacity_grav': 'Gravimetric capacity',
    'capacity_vol': 'Volumetric capacity',
    'energy_grav': 'Gravimetric energy',
    'energy_vol': 'Volumetric energy',
    'max_delta_volume': 'Maximum volume change',
    'stability_charge': 'Charged-state stability',
    'stability_discharge': 'Discharged-state stability',
    'stability_worst': 'Worst endpoint stability',
}
m['target'] = pd.Categorical(m['target'], categories=order, ordered=True)
m = m.sort_values('target').reset_index(drop=True)

# Bundled font avoids system-font dependence and embeds cleanly in PDF/SVG.
font_path = Path(mpl.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'
font_manager.fontManager.addfont(str(font_path))
PUBLICATION_FONT = font_manager.FontProperties(fname=str(font_path)).get_name()

mpl.rcParams.update({
    'font.family': PUBLICATION_FONT,
    'font.size': 7.6,
    'axes.labelsize': 8.3,
    'axes.titlesize': 8.4,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.75,
    'lines.linewidth': 0.9,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.transparent': False,
    'mathtext.fontset': 'dejavusans',
})

BLUE = '#0072B2'
ORANGE = '#E69F00'
GREEN = '#009E73'
VERM = '#D55E00'
PURPLE = '#CC79A7'
SKY = '#56B4E9'
LIGHT = '#F5F5F5'
DARK = '#222222'
MID = '#666666'
GRID = '#D9D9D9'


def mm(x):
    return x / 25.4


def save_all(fig, stem, tiff_dpi=600):
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')

    # Matplotlib may emit CRLF in SVG files on Windows.  Normalize the SVG
    # bytes to LF before hashing so Git checkout/normalization cannot make the
    # checksum manifest stale on another platform.
    svg_path = OUT / f'{stem}.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    svg_bytes = svg_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    svg_path.write_bytes(svg_bytes)

    fig.savefig(OUT / f'{stem}_preview.png', dpi=300, bbox_inches='tight')
    tiff_path = OUT / f'{stem}.tif'
    try:
        fig.savefig(
            tiff_path, dpi=tiff_dpi, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'}
        )
    except Exception as exc:
        # Some older Windows/Pillow builds lack TIFF-LZW support. The uncompressed
        # TIFF is scientifically identical and still suitable for journal upload.
        print(f'WARNING: TIFF-LZW failed for {stem}: {exc}')
        print('         Retrying as uncompressed TIFF.')
        fig.savefig(tiff_path, dpi=tiff_dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: target-specific provenance and stage eligibility
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(mm(190), mm(154)))
gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.16], hspace=0.30)

ax = fig.add_subplot(gs[0])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.0, 0.99, '(a)', fontsize=9.5, fontweight='bold', va='top')

# Provenance card: stacked text prevents cross-column collisions on Windows.
hdr = FancyBboxPatch(
    (0.055, 0.685), 0.89, 0.265,
    boxstyle='round,pad=0.012,rounding_size=0.018',
    fc=LIGHT, ec=MID, lw=0.75
)
ax.add_patch(hdr)
ax.text(
    0.082, 0.875, 'Target-specific descriptor eligibility',
    fontweight='bold', fontsize=8.45, va='center'
)
ax.text(
    0.082, 0.815, 'Physical dependency graph + rule-based compiler',
    fontsize=7.0, va='center'
)
ax.text(
    0.50, 0.752,
    '270 candidate descriptors × 9 targets   |   2,430 feature-target classifications',
    fontsize=6.15, va='center', ha='center'
)
ax.text(
    0.50, 0.708,
    '291 nodes; 2,438 directed relations   |   compiler/reference agreement: 2,430/2,430',
    fontsize=6.15, va='center', ha='center'
)

xs = [0.055, 0.3725, 0.690]
width = 0.255
stage_specs = [
    ('P1', 'Composition',
     'Framework composition\nand working-ion\ndescriptors',
     '73 descriptors', '#EAF2F8'),
    ('P2', 'Relaxed structure',
     'Adds target-eligible\ndescriptors from\nDFT-relaxed structures',
     '77-93 descriptors', '#EAF5EF'),
    ('P3', 'Later computed\ninformation',
     'Adds eligible energetic,\nstability, and electronic\ndescriptors',
     '96-112 descriptors', '#FFF3E0'),
]

for k, (x, spec) in enumerate(zip(xs, stage_specs)):
    p, title, body, foot, fc = spec
    box = FancyBboxPatch(
        (x, 0.075), width, 0.50,
        boxstyle='round,pad=0.014,rounding_size=0.018',
        fc=fc, ec=DARK, lw=0.75
    )
    ax.add_patch(box)
    ax.text(x + 0.018, 0.535, p, fontweight='bold', fontsize=9.0, va='top')
    ax.text(
        x + 0.018, 0.445, title,
        fontweight='bold', fontsize=7.5, va='top', linespacing=1.08
    )
    ax.text(
        x + 0.018, 0.295, body,
        fontsize=6.45, va='top', linespacing=1.15
    )
    ax.text(
        x + 0.018, 0.108, foot,
        fontsize=6.9, fontweight='bold', va='bottom'
    )
    if k < 2:
        ax.add_patch(FancyArrowPatch(
            (x + width + 0.010, 0.33), (xs[k + 1] - 0.010, 0.33),
            arrowstyle='-|>', mutation_scale=10, lw=0.85, color=MID
        ))

ax2 = fig.add_subplot(gs[1])
ax2.text(
    -0.19, 1.04, '(b)', transform=ax2.transAxes,
    fontsize=9.5, fontweight='bold', va='top'
)
counts = m[['n_features_P1', 'n_features_P2', 'n_features_P3']].to_numpy(float)
im = ax2.imshow(counts, cmap='viridis', aspect='auto', vmin=70, vmax=115)
ax2.set_xticks([0, 1, 2], ['P1', 'P2', 'P3'])
ax2.set_yticks(np.arange(len(order)), [labels[t] for t in order])
ax2.set_xlabel('Computational workflow stage')
for i in range(counts.shape[0]):
    for j in range(3):
        v = counts[i, j]
        norm_v = (v - 70) / 45
        txt_color = 'white' if norm_v < 0.45 else DARK
        ax2.text(j, i, f'{int(v)}', ha='center', va='center', fontsize=7.0, color=txt_color)
for s in ax2.spines.values():
    s.set_visible(False)
ax2.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, 9, 1), minor=True)
ax2.grid(which='minor', color='white', linewidth=1.1)
ax2.tick_params(which='minor', bottom=False, left=False)
cbar = fig.colorbar(im, ax=ax2, fraction=0.026, pad=0.02)
cbar.ax.set_title('Descriptors', fontsize=6.4, pad=3)
cbar.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.285, right=0.935, top=0.97, bottom=0.07)
save_all(fig, 'Fig1_information_provenance_and_eligibility_submission')


# ---------------------------------------------------------------------------
# Figure 2: earliest useful stage / skill heatmaps
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(mm(190), mm(118)), sharey=True)
stages = ['P1', 'P2', 'P3']
norm = TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.75)
for a, metric, panel in zip(axs, ['MAE', 'RMSE'], ['(a)', '(b)']):
    vals = np.column_stack([m[f'{metric}_skill_{s}'].to_numpy(float) for s in stages])
    im = a.imshow(vals, cmap='RdBu', norm=norm, aspect='auto')
    a.set_xticks(range(3), stages)
    a.set_xlabel('Computational workflow stage')
    a.set_yticks(range(9), [labels[t] for t in order])

    # Panel tag and title use a common baseline with explicit separation.
    a.text(
        -0.14, 1.075, panel, transform=a.transAxes,
        fontsize=9.5, fontweight='bold', va='bottom', ha='left'
    )
    a.text(
        0.50, 1.075, f'{metric} skill vs mean predictor', transform=a.transAxes,
        fontsize=8.4, va='bottom', ha='center'
    )

    for i in range(9):
        for j in range(3):
            v = vals[i, j]
            a.text(
                j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.4,
                color=('white' if abs(v) > 0.43 else DARK)
            )
    for i, row in m.iterrows():
        e = row['primary_ExtraTrees_earliest_useful_stage_10pct']
        if e in stages:
            j = stages.index(e)
            a.add_patch(Rectangle(
                (j - 0.47, i - 0.47), 0.94, 0.94,
                fill=False, ec=DARK, lw=1.45
            ))
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    a.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    a.grid(which='minor', color='white', linewidth=1.1)
    a.tick_params(which='minor', bottom=False, left=False)

cax = fig.add_axes([0.905, 0.19, 0.018, 0.64])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Relative error reduction vs mean predictor')
cb.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.25, right=0.875, top=0.81, bottom=0.12, wspace=0.10)
save_all(fig, 'Fig2_stage_dependent_learnability_submission')


# ---------------------------------------------------------------------------
# Figure 3: paired marginal information value
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(mm(190), mm(130)), sharey=True)
transitions = [
    ('P1_to_P2', 'P1 to P2', 'o', BLUE, -0.18),
    ('P2_to_P3', 'P2 to P3', 's', ORANGE, 0.0),
    ('P1_to_P3', 'P1 to P3', 'D', GREEN, 0.18),
]
y = np.arange(9)
for a, metric, panel in zip(axs, ['mae', 'rmse'], ['(a)', '(b)']):
    for key, name, marker, color, off in transitions:
        x = m[f'{key}_relative_{metric}_improvement'].to_numpy(float)
        lo = m[f'{key}_relative_{metric}_ci_low'].to_numpy(float)
        hi = m[f'{key}_relative_{metric}_ci_high'].to_numpy(float)
        a.errorbar(
            x, y + off,
            xerr=np.vstack([x - lo, hi - x]),
            fmt=marker, ms=3.8, mew=0.6, mfc=color, mec=color,
            ecolor=color, elinewidth=0.8, capsize=1.8,
            label=name, zorder=3
        )
    a.axvline(0, color=MID, lw=0.8, ls='--', zorder=1)
    a.set_xlim(-0.20, 0.68)
    a.set_xlabel(f'Relative {metric.upper()} improvement')
    a.set_yticks(y, [labels[t] for t in order])
    a.grid(axis='x', color=GRID, lw=0.5)
    a.text(-0.13, 1.04, panel, transform=a.transAxes, fontsize=9.5, fontweight='bold', va='top')
    a.set_title(f'{metric.upper()} change', pad=5)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
axs[0].invert_yaxis()
axs[0].legend(frameon=False, loc='lower right', handlelength=1.4, borderaxespad=0.3)
fig.subplots_adjust(left=0.26, right=0.98, top=0.90, bottom=0.12, wspace=0.14)
save_all(fig, 'Fig3_incremental_information_value_submission')


# ---------------------------------------------------------------------------
# Figure 4: chemical-domain robustness
# ---------------------------------------------------------------------------
split_order = [
    ('framework', 'Framework formula'),
    ('leave_chemical_system_out', 'Host chemical system'),
    ('leave_family_out', 'Coarse chemistry family'),
    ('leave_working_ion_out', 'Working ion'),
]
vals_by_metric = {}
for metric in ['mae', 'rmse']:
    mat = np.full((9, 4), np.nan)
    mat[:, 0] = m[f'P1_to_P3_relative_{metric}_improvement'].to_numpy(float)
    for j, (split, _) in enumerate(split_order[1:], start=1):
        sub = d[d['split_name'] == split]
        for i, t in enumerate(order):
            ss = sub[sub['target'] == t].set_index('stage')
            p1 = float(ss.loc['P1', f'macro_et_{metric}'])
            p3 = float(ss.loc['P3', f'macro_et_{metric}'])
            mat[i, j] = (p1 - p3) / p1
    vals_by_metric[metric] = mat

fig, axs = plt.subplots(1, 2, figsize=(mm(190), mm(123)), sharey=True)
norm = TwoSlopeNorm(vmin=-0.18, vcenter=0.0, vmax=0.62)
for a, metric, panel in zip(axs, ['mae', 'rmse'], ['(a)', '(b)']):
    vals = vals_by_metric[metric]
    im = a.imshow(vals, cmap='RdBu', norm=norm, aspect='auto')
    a.set_xticks(range(4), [x[1] for x in split_order], rotation=22, ha='right')
    a.set_yticks(range(9), [labels[t] for t in order])

    # Shorter title and separated panel tag prevent overlap on Windows.
    a.text(
        -0.14, 1.085, panel, transform=a.transAxes,
        fontsize=9.5, fontweight='bold', va='bottom', ha='left'
    )
    a.text(
        0.50, 1.085, f'P1 to P3 relative {metric.upper()} improvement',
        transform=a.transAxes, fontsize=7.9, va='bottom', ha='center'
    )

    for i in range(9):
        for j in range(4):
            v = vals[i, j]
            a.text(
                j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.2,
                color=('white' if abs(v) > 0.36 else DARK)
            )
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    a.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    a.grid(which='minor', color='white', linewidth=1.05)
    a.tick_params(which='minor', bottom=False, left=False)

cax = fig.add_axes([0.905, 0.20, 0.018, 0.62])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Relative error reduction (positive = lower P3 error)')
cb.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.25, right=0.875, top=0.79, bottom=0.25, wspace=0.12)
save_all(fig, 'Fig4_chemical_domain_robustness_submission')


# ---------------------------------------------------------------------------
# Figure 5: composition-space applicability-domain diagnostic
# ---------------------------------------------------------------------------
target_order = ['average_voltage', 'capacity_grav', 'energy_grav', 'max_delta_volume', 'stability_worst']
target_labels = {
    'average_voltage': 'Average voltage',
    'capacity_grav': 'Gravimetric capacity',
    'energy_grav': 'Gravimetric energy',
    'max_delta_volume': 'Maximum volume change',
    'stability_worst': 'Worst endpoint stability',
}
split_specs = [
    ('random_split', 'Random', 'o', BLUE),
    ('framework_groupkfold', 'Framework formula', 's', ORANGE),
    ('leave_chemical_system_out', 'Host chemical system', '^', GREEN),
    ('leave_family_out', 'Coarse chemistry family', 'D', VERM),
    ('leave_working_ion_out', 'Working ion', 'P', PURPLE),
]
fig, ax = plt.subplots(figsize=(mm(190), mm(112)))
y = np.arange(len(target_order))
for split, lab, marker, color in split_specs:
    sub = ad[ad['split_name'].eq(split)].set_index('target').reindex(target_order)
    ax.scatter(
        sub['out_to_in_domain_mae_ratio'], y,
        s=30, marker=marker, facecolors=color, edgecolors='white', linewidths=0.4,
        label=lab, zorder=3
    )
ax.axvline(1.0, color='0.35', lw=0.9, ls='--', zorder=1)
ax.set_xscale('log')
ax.set_xlim(0.6, 6.2)
ax.set_xticks([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks(y, [target_labels[t] for t in target_order])
ax.invert_yaxis()
ax.set_xlabel('Out-of-domain MAE / in-domain MAE')
ax.grid(axis='x', which='major', color=GRID, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12),
          handletextpad=0.4, columnspacing=1.2)
fig.subplots_adjust(left=0.29, right=0.98, top=0.82, bottom=0.15)
save_all(fig, 'Fig5_applicability_domain_diagnostic_submission')


# ---------------------------------------------------------------------------
# Figure 6: sodium screening + first-principles handoff
# Legend is placed in the figure margin so it cannot cover data or annotations.
# ---------------------------------------------------------------------------
selected = cand.loc[cand['battery_formula'].eq('Na1-3CoPCO7')].iloc[0]
sel_v = dft.loc[dft['set'].eq('selected_candidate')].iloc[0]
ref_v = dft.loc[dft['set'].eq('denser_reference')].iloc[0]
score_min = cand['final_triage_score'].min()
score_max = cand['final_triage_score'].max()
den = max(score_max - score_min, 1e-12)
sizes = 18 + 120 * (cand['final_triage_score'] - score_min) / den

fig, ax = plt.subplots(figsize=(mm(190), mm(124)))
ax.scatter(
    cand['stability_worst'], cand['energy_grav'],
    s=sizes, alpha=0.30, linewidths=0.40,
    facecolors=SKY, edgecolors=BLUE,
    label='Screen-passing Na candidates', zorder=2
)
ax.scatter(
    [selected['stability_worst']], [selected['energy_grav']],
    s=145, marker='*', facecolors=ORANGE, edgecolors=VERM, linewidths=0.75,
    label='DFT handoff candidate', zorder=6
)

ax.set_xlabel(r'Worst endpoint stability (eV atom$^{-1}$)')
ax.set_ylabel(r'Gravimetric energy (Wh kg$^{-1}$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.margins(x=0.06, y=0.08)

xmin, xmax = cand['stability_worst'].min(), cand['stability_worst'].max()
ymin, ymax = cand['energy_grav'].min(), cand['energy_grav'].max()
xspan = max(xmax - xmin, 1e-12)
yspan = max(ymax - ymin, 1e-12)

# Data-coordinate annotation placement adapts if the candidate is near an edge.
text_x = selected['stability_worst'] + 0.08 * xspan
text_y = selected['energy_grav'] + 0.11 * yspan
ha = 'left'
va = 'bottom'
if text_x > xmax - 0.10 * xspan:
    text_x = selected['stability_worst'] - 0.20 * xspan
    ha = 'right'
if text_y > ymax - 0.08 * yspan:
    text_y = selected['energy_grav'] - 0.12 * yspan
    va = 'top'

ax.annotate(
    r'NaCoPO$_4$CO$_3$ $\rightarrow$ Na$_3$CoPO$_4$CO$_3$',
    xy=(selected['stability_worst'], selected['energy_grav']),
    xytext=(text_x, text_y), textcoords='data',
    ha=ha, va=va, fontsize=7.0,
    arrowprops=dict(arrowstyle='->', lw=0.8, color='0.25'),
    bbox=dict(
        boxstyle='round,pad=0.26', facecolor='white',
        edgecolor='0.65', linewidth=0.6, alpha=0.92
    ),
    zorder=7
)

# Legend in the top figure margin: no collision with data, star, or annotation.
handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(
    handles, legend_labels,
    loc='upper left', bbox_to_anchor=(0.12, 0.985),
    ncol=2, frameon=False,
    handlelength=1.8, handletextpad=0.75,
    columnspacing=1.6, markerscale=0.85,
    borderaxespad=0.0
)

info = (
    'Fixed-geometry PBE handoff\n'
    f"selected mesh: {sel_v['average_voltage_V']:.6f} V\n"
    f"denser mesh: {ref_v['average_voltage_V']:.6f} V\n"
    f"|ΔV| = {ref_v['abs_delta_voltage_from_selected_V'] * 1000:.3f} mV"
)
fig.text(
    0.975, 0.975, info,
    ha='right', va='top', fontsize=7.0,
    bbox=dict(
        boxstyle='round,pad=0.35', facecolor='white',
        edgecolor='0.55', linewidth=0.65, alpha=0.95
    )
)
fig.subplots_adjust(left=0.13, right=0.98, top=0.79, bottom=0.14)
save_all(fig, 'Fig6_sodium_triage_dft_handoff_submission')


run_manifest = {
    'python': sys.version,
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'matplotlib': mpl.__version__,
    'matplotlib_backend': str(mpl.get_backend()),
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'repository_root': str(ROOT),
    'output_directory': str(OUT),
    'source_files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in required},
}
# Write the run manifest with explicit LF bytes for cross-platform stability.
(OUT / 'submission_artwork_run_manifest.json').write_bytes(
    (json.dumps(run_manifest, indent=2) + '\n').encode('utf-8')
)

# Hash only the final artwork assets.  The checksum CSV and environment/run
# manifest are metadata and are intentionally excluded from their own asset
# checksum set.
records = []
for p in sorted(OUT.iterdir()):
    if (
        p.is_file()
        and p.name not in {
            'submission_artwork_sha256.csv',
            'submission_artwork_run_manifest.json',
        }
    ):
        records.append({
            'file': p.name,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'bytes': p.stat().st_size,
        })
pd.DataFrame(records).to_csv(OUT / 'submission_artwork_sha256.csv', index=False)

print('Submission artwork generated from frozen source data.')
print(f'Output directory: {OUT}')
print('Files generated:')
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(f'  {p.name}')
