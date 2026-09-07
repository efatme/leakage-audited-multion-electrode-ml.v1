from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'papers' / 'cmt' / 'publication' / 'outputs'
SRC = ROOT / 'figures' / 'source_data'
REF = ROOT / 'papers' / 'cmt' / 'publication' / 'reference_rendering'
OUTSRC = OUT / 'source_data'
REF.mkdir(parents=True, exist_ok=True)
OUTSRC.mkdir(parents=True, exist_ok=True)

# Publication defaults: restrained journal styling, no large in-artwork titles.
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.labelsize': 8.2,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

WIDTH_IN = 190 / 25.4

def save_all(fig, stem, height_in):
    fig.set_size_inches(WIDTH_IN, height_in)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.tif', dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / f'{stem}_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------- Figure 5: applicability-domain diagnostic ----------
ad = pd.read_csv(SRC / 'Figure_5_applicability_domain_source.csv')
ad.to_csv(OUTSRC / 'Fig5_applicability_domain_source.csv', index=False)

target_order = [
    'average_voltage', 'capacity_grav', 'energy_grav', 'max_delta_volume', 'stability_worst'
]
target_labels = {
    'average_voltage': 'Average voltage',
    'capacity_grav': 'Gravimetric capacity',
    'energy_grav': 'Gravimetric energy',
    'max_delta_volume': 'Maximum volume change',
    'stability_worst': 'Worst endpoint stability',
}
split_order = [
    'random_split', 'framework_groupkfold', 'leave_chemical_system_out',
    'leave_family_out', 'leave_working_ion_out'
]
split_labels = {
    'random_split': 'Random',
    'framework_groupkfold': 'Framework formula',
    'leave_chemical_system_out': 'Host chemical system',
    'leave_family_out': 'Coarse chemistry family',
    'leave_working_ion_out': 'Working ion',
}

fig, ax = plt.subplots()
y = np.arange(len(target_order))
for split in split_order:
    sub = ad[ad['split_name'].eq(split)].set_index('target').reindex(target_order)
    ax.scatter(sub['out_to_in_domain_mae_ratio'], y, s=28, label=split_labels[split], zorder=3)

ax.axvline(1.0, color='0.35', lw=1.0, ls='--', zorder=1)
ax.set_xscale('log')
ax.set_xlim(0.6, 6.2)
ax.set_xticks([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks(y, [target_labels[t] for t in target_order])
ax.invert_yaxis()
ax.set_xlabel('Out-of-domain MAE / in-domain MAE')
ax.grid(axis='x', which='major', alpha=0.18, lw=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.11), handletextpad=0.4, columnspacing=1.2)
fig.subplots_adjust(left=0.29, right=0.98, top=0.83, bottom=0.14)
save_all(fig, 'Fig5_applicability_domain_diagnostic', 4.75)

# ---------- Figure 6: sodium triage + DFT handoff ----------
cand = pd.read_csv(SRC / 'Figure_6_sodium_candidates_source.csv')
dft = pd.read_csv(SRC / 'Figure_6_dft_voltage_source.csv')
cand.to_csv(OUTSRC / 'Fig6_sodium_candidates_source.csv', index=False)
dft.to_csv(OUTSRC / 'Fig6_dft_voltage_source.csv', index=False)
selected = cand.loc[cand['battery_formula'].eq('Na1-3CoPCO7')].iloc[0]
sel_v = dft.loc[dft['set'].eq('selected_candidate')].iloc[0]
ref_v = dft.loc[dft['set'].eq('denser_reference')].iloc[0]
score_min = cand['final_triage_score'].min()
score_max = cand['final_triage_score'].max()
den = max(score_max - score_min, 1e-12)
sizes = 18 + 120 * (cand['final_triage_score'] - score_min) / den

fig, ax = plt.subplots()
ax.scatter(cand['stability_worst'], cand['energy_grav'], s=sizes, alpha=0.35,
           linewidths=0.45, label='Screen-passing Na candidates')
ax.scatter([selected['stability_worst']], [selected['energy_grav']], s=140, marker='*',
           linewidths=0.8, label='DFT handoff candidate', zorder=5)
ax.annotate(r'NaCoPO$_4$CO$_3$ $\rightarrow$ Na$_3$CoPO$_4$CO$_3$',
            xy=(selected['stability_worst'], selected['energy_grav']),
            xytext=(24, 12), textcoords='offset points', ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', lw=0.8, color='0.25'))
ax.set_xlabel(r'Worst endpoint stability (eV atom$^{-1}$)')
ax.set_ylabel(r'Gravimetric energy (Wh kg$^{-1}$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.legend(frameon=False, loc='lower left')
info = (
    'Fixed-geometry PBE handoff\n'
    f'selected mesh: {sel_v["average_voltage_V"]:.6f} V\n'
    f'dense reference: {ref_v["average_voltage_V"]:.6f} V\n'
    f'|ΔV| = {ref_v["abs_delta_voltage_from_selected_V"]:.6f} V'
)
ax.text(0.985, 0.965, info, transform=ax.transAxes, ha='right', va='top', fontsize=7.0,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.55', linewidth=0.7, alpha=0.95))
fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.14)
save_all(fig, 'Fig6_sodium_triage_dft_handoff', 4.95)

print('Generated Fig5 and Fig6 from exact repository source CSVs.')
