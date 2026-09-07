from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / 'papers' / 'cmt' / 'results'
OUT = ROOT / 'papers' / 'cmt' / 'publication' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

MASTER = RES / 'stage_f' / 'stage_f_master_result_matrix.csv'
DOMAIN = RES / 'stage_d' / 'stage_d_domain_summary.csv'
for p in [MASTER, DOMAIN]:
    if not p.exists():
        raise FileNotFoundError(p)

m = pd.read_csv(MASTER)
d = pd.read_csv(DOMAIN)

order = [
    'average_voltage','capacity_grav','capacity_vol','energy_grav','energy_vol',
    'max_delta_volume','stability_charge','stability_discharge','stability_worst'
]
labels = {
    'average_voltage':'Average voltage',
    'capacity_grav':'Gravimetric capacity',
    'capacity_vol':'Volumetric capacity',
    'energy_grav':'Gravimetric energy',
    'energy_vol':'Volumetric energy',
    'max_delta_volume':'Maximum volume change',
    'stability_charge':'Charged-state stability',
    'stability_discharge':'Discharged-state stability',
    'stability_worst':'Worst endpoint stability',
}
m['target'] = pd.Categorical(m['target'], categories=order, ordered=True)
m = m.sort_values('target').reset_index(drop=True)

# Elsevier-compatible typography and vector embedding.
# Use Matplotlib's bundled DejaVu Sans explicitly so the publication script
# does not depend on system font discovery (important for Windows/conda servers).
_BUNDLED_FONT = Path(mpl.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'
if not _BUNDLED_FONT.exists():
    raise FileNotFoundError(
        f'Required bundled publication font not found: {_BUNDLED_FONT}. '
        'The Matplotlib installation is incomplete.'
    )
font_manager.fontManager.addfont(str(_BUNDLED_FONT))
_PUBLICATION_FONT = font_manager.FontProperties(fname=str(_BUNDLED_FONT)).get_name()

mpl.rcParams.update({
    'font.family': _PUBLICATION_FONT,
    'font.size': 7.5,
    'axes.labelsize': 8.0,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.7,
    'lines.linewidth': 0.9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.transparent': False,
})

# Colorblind-aware palette; all quantitative panels also carry numeric labels / marker-shape redundancy.
BLUE = '#0072B2'; ORANGE = '#E69F00'; GREEN = '#009E73'; VERM = '#D55E00'; PURPLE = '#CC79A7'
LIGHT = '#F4F4F4'; DARK = '#222222'; MID = '#666666'; GRID = '#D9D9D9'


def mm(x): return x / 25.4

def save_all(fig, stem, tiff_dpi=600):
    fig.savefig(OUT / f'{stem}.pdf')
    fig.savefig(OUT / f'{stem}.svg')
    fig.savefig(OUT / f'{stem}_preview.png', dpi=300)
    fig.savefig(OUT / f'{stem}.tif', dpi=tiff_dpi, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def annotate_heatmap(ax, vals, fmt='.2f', threshold=None):
    finite = vals[np.isfinite(vals)]
    vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i,j]
            if not np.isfinite(v): continue
            color = 'white' if abs(v) > 0.55*vmax else DARK
            ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=6.5, color=color)
            if threshold is not None and v >= threshold:
                ax.add_patch(Rectangle((j-0.49,i-0.49),0.98,0.98,fill=False,ec=DARK,lw=1.1))

# FIGURE 1 -----------------------------------------------------------------
fig = plt.figure(figsize=(mm(190), mm(145)))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.25], hspace=0.30)
ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.0, 0.98, '(a)', fontsize=9.5, fontweight='bold', va='top')
# Header strip
hdr = FancyBboxPatch((0.08,0.72),0.84,0.23,boxstyle='round,pad=0.012,rounding_size=0.018',
                     fc=LIGHT,ec=MID,lw=0.8)
ax.add_patch(hdr)
ax.text(0.10,0.875,'Target-specific descriptor eligibility',fontweight='bold',fontsize=8.2,va='center')
ax.text(0.10,0.805,'Expert dependency graph + rule-based compiler',fontsize=7.0,va='center')
ax.text(0.54,0.885,'270 candidate features x 9 targets',fontsize=6.8,va='center')
ax.text(0.54,0.835,'2,430 target-feature classifications',fontsize=6.8,va='center')
ax.text(0.54,0.785,'291 nodes; 2,438 directed relations',fontsize=6.8,va='center')
ax.text(0.54,0.735,'compiler/reference match: 2,430/2,430',fontsize=6.8,va='center')

xs=[0.08,0.375,0.67]; widths=[0.245]*3
stage_txt=[
    ('P1','Composition','68 framework-composition\n5 working-ion descriptors','73 total features'),
    ('P2','Relaxed structure','Adds target-eligible descriptors\nfrom DFT-relaxed structures','77-93 total features'),
    ('P3','Legitimate post-DFT','Adds target-eligible energetic,\nstability and electronic descriptors','96-112 total features'),
]
colors=['#E8F1F8','#E8F5EF','#FFF3E0']
for k,(x,w,txt,fc) in enumerate(zip(xs,widths,stage_txt,colors)):
    box=FancyBboxPatch((x,0.10),w,0.48,boxstyle='round,pad=0.014,rounding_size=0.018',fc=fc,ec=DARK,lw=0.8)
    ax.add_patch(box)
    ax.text(x+0.018,0.53,txt[0],fontweight='bold',fontsize=9,va='top')
    ax.text(x+0.018,0.44,txt[1],fontweight='bold',fontsize=7.7,va='top')
    ax.text(x+0.018,0.31,txt[2],fontsize=6.6,va='top',linespacing=1.18)
    ax.text(x+0.018,0.145,txt[3],fontsize=7.0,va='bottom',fontweight='bold')
    if k<2:
        ax.add_patch(FancyArrowPatch((x+w+0.012,0.34),(xs[k+1]-0.012,0.34),arrowstyle='-|>',mutation_scale=10,lw=0.9,color=MID))

ax2=fig.add_subplot(gs[1])
ax2.text(-0.13,1.07,'(b)',transform=ax2.transAxes,fontsize=9.5,fontweight='bold',va='top')
counts=m[['n_features_P1','n_features_P2','n_features_P3']].to_numpy(float)
im=ax2.imshow(counts,cmap='Blues',aspect='auto',vmin=70,vmax=115)
ax2.set_xticks([0,1,2],['P1','P2','P3'])
ax2.set_yticks(np.arange(len(order)),[labels[t] for t in order])
ax2.set_xlabel('Information stage')
for i in range(counts.shape[0]):
    for j in range(3): ax2.text(j,i,f'{int(counts[i,j])}',ha='center',va='center',fontsize=7,color=DARK)
for s in ax2.spines.values(): s.set_visible(False)
ax2.set_xticks(np.arange(-.5,3,1),minor=True); ax2.set_yticks(np.arange(-.5,9,1),minor=True)
ax2.grid(which='minor',color='white',linewidth=1.2); ax2.tick_params(which='minor',bottom=False,left=False)
cbar=fig.colorbar(im,ax=ax2,fraction=0.026,pad=0.02); cbar.ax.set_title('Features',fontsize=6.5,pad=3); cbar.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.23,right=0.94,top=0.97,bottom=0.08)
save_all(fig,'Fig1_information_provenance_and_eligibility')

# FIGURE 2 -----------------------------------------------------------------
fig,axs=plt.subplots(1,2,figsize=(mm(190),mm(115)),sharey=True)
stages=['P1','P2','P3']
norm=TwoSlopeNorm(vmin=-0.22,vcenter=0.0,vmax=0.75)
for a,metric,panel in zip(axs,['MAE','RMSE'],['(a)','(b)']):
    vals=np.column_stack([m[f'{metric}_skill_{s}'].to_numpy(float) for s in stages])
    im=a.imshow(vals,cmap='RdBu',norm=norm,aspect='auto')
    a.set_xticks(range(3),stages); a.set_xlabel('Information stage')
    a.set_yticks(range(9),[labels[t] for t in order])
    a.text(-0.13,1.04,panel,transform=a.transAxes,fontsize=9.5,fontweight='bold',va='top')
    a.set_title(f'{metric} skill vs DummyMean',fontsize=8.2,pad=5)
    for i in range(9):
        for j in range(3):
            v=vals[i,j]
            a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.4,color=('white' if abs(v)>0.43 else DARK))
    # Outline earliest useful cell from combined MAE+RMSE rule.
    for i,row in m.iterrows():
        e=row['primary_ExtraTrees_earliest_useful_stage_10pct']
        if e in stages:
            j=stages.index(e); a.add_patch(Rectangle((j-0.47,i-0.47),0.94,0.94,fill=False,ec=DARK,lw=1.5))
    for s in a.spines.values(): s.set_visible(False)
    a.set_xticks(np.arange(-.5,3,1),minor=True); a.set_yticks(np.arange(-.5,9,1),minor=True)
    a.grid(which='minor',color='white',linewidth=1.2); a.tick_params(which='minor',bottom=False,left=False)
cax=fig.add_axes([0.91,0.20,0.018,0.62]); cb=fig.colorbar(im,cax=cax); cb.set_label('Relative error reduction vs DummyMean'); cb.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.25,right=0.88,top=0.89,bottom=0.12,wspace=0.08)
save_all(fig,'Fig2_stage_dependent_learnability')

# FIGURE 3 -----------------------------------------------------------------
fig,axs=plt.subplots(1,2,figsize=(mm(190),mm(132)),sharey=True)
transitions=[('P1_to_P2','P1 to P2','o',BLUE,-0.18),('P2_to_P3','P2 to P3','s',ORANGE,0.0),('P1_to_P3','P1 to P3','D',GREEN,0.18)]
y=np.arange(9)
for a,metric,panel in zip(axs,['mae','rmse'],['(a)','(b)']):
    for key,name,marker,color,off in transitions:
        x=m[f'{key}_relative_{metric}_improvement'].to_numpy(float)
        lo=m[f'{key}_relative_{metric}_ci_low'].to_numpy(float)
        hi=m[f'{key}_relative_{metric}_ci_high'].to_numpy(float)
        a.errorbar(x,y+off,xerr=np.vstack([x-lo,hi-x]),fmt=marker,ms=3.7,mew=0.6,mfc=color,mec=color,
                   ecolor=color,elinewidth=0.8,capsize=1.8,label=name,zorder=3)
    a.axvline(0,color=MID,lw=0.8,ls='--',zorder=1)
    a.set_xlim(-0.20,0.68); a.set_xlabel(f'Relative {metric.upper()} improvement')
    a.set_yticks(y,[labels[t] for t in order]); a.grid(axis='x',color=GRID,lw=0.5)
    a.text(-0.13,1.04,panel,transform=a.transAxes,fontsize=9.5,fontweight='bold',va='top')
    a.set_title(f'{metric.upper()} change',fontsize=8.2,pad=5)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
axs[0].invert_yaxis()
axs[0].legend(frameon=False,loc='lower right',handlelength=1.4,borderaxespad=0.3)
fig.subplots_adjust(left=0.26,right=0.98,top=0.90,bottom=0.12,wspace=0.14)
save_all(fig,'Fig3_incremental_information_value')

# FIGURE 4 -----------------------------------------------------------------
split_order=[
    ('framework','Framework formula'),
    ('leave_chemical_system_out','Host chemical system'),
    ('leave_family_out','Coarse chemistry family'),
    ('leave_working_ion_out','Working ion'),
]
vals_by_metric={}
for metric in ['mae','rmse']:
    mat=np.full((9,4),np.nan)
    mat[:,0]=m[f'P1_to_P3_relative_{metric}_improvement'].to_numpy(float)
    for j,(split,_) in enumerate(split_order[1:],start=1):
        sub=d[d['split_name']==split]
        for i,t in enumerate(order):
            ss=sub[sub['target']==t].set_index('stage')
            p1=float(ss.loc['P1',f'macro_et_{metric}']); p3=float(ss.loc['P3',f'macro_et_{metric}'])
            mat[i,j]=(p1-p3)/p1
    vals_by_metric[metric]=mat
fig,axs=plt.subplots(1,2,figsize=(mm(190),mm(118)),sharey=True)
norm=TwoSlopeNorm(vmin=-0.18,vcenter=0.0,vmax=0.62)
for a,metric,panel in zip(axs,['mae','rmse'],['(a)','(b)']):
    vals=vals_by_metric[metric]
    im=a.imshow(vals,cmap='RdBu',norm=norm,aspect='auto')
    a.set_xticks(range(4),[x[1] for x in split_order],rotation=26,ha='right')
    a.set_yticks(range(9),[labels[t] for t in order])
    a.set_title(f'P1 to P3 relative {metric.upper()} improvement',fontsize=8.1,pad=5)
    a.text(-0.13,1.04,panel,transform=a.transAxes,fontsize=9.5,fontweight='bold',va='top')
    for i in range(9):
        for j in range(4):
            v=vals[i,j]; a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.2,color=('white' if abs(v)>0.36 else DARK))
            if v>0: a.add_patch(Rectangle((j-0.48,i-0.48),0.96,0.96,fill=False,ec=DARK,lw=0.55))
    for s in a.spines.values(): s.set_visible(False)
    a.set_xticks(np.arange(-.5,4,1),minor=True); a.set_yticks(np.arange(-.5,9,1),minor=True)
    a.grid(which='minor',color='white',linewidth=1.1); a.tick_params(which='minor',bottom=False,left=False)
cax=fig.add_axes([0.91,0.20,0.018,0.62]); cb=fig.colorbar(im,cax=cax); cb.set_label('Relative error reduction (positive favors P3)'); cb.outline.set_linewidth(0.6)
fig.subplots_adjust(left=0.25,right=0.88,top=0.89,bottom=0.22,wspace=0.08)
save_all(fig,'Fig4_chemical_domain_robustness')

# TABLE 1 ------------------------------------------------------------------
rows=[]
for _,r in m.iterrows():
    t=str(r['target'])
    threshold_stable=(r['earliest_stage_5pct']==r['earliest_stage_10pct']==r['earliest_stage_20pct'])
    domain_both=sum(bool(r[x]) for x in ['host_system_P1_to_P3_macro_both_metrics_improve','coarse_family_P1_to_P3_macro_both_metrics_improve','working_ion_P1_to_P3_macro_both_metrics_improve'])
    primary=r['primary_ExtraTrees_earliest_useful_stage_10pct']
    primary_display='None through P3' if primary=='none_through_P3' else primary
    rows.append({
        'Target':labels[t],
        'P1 features':int(r['n_features_P1']),
        'P2 features':int(r['n_features_P2']),
        'P3 features':int(r['n_features_P3']),
        'Primary earliest useful stage':primary_display,
        '5/10/20% threshold assignment':'stable' if threshold_stable else 'sensitive',
        'Ridge exact-stage sensitivity':'same' if bool(r['ExtraTrees_Ridge_same_earliest_stage']) else 'different',
        'Domain tests with P1-to-P3 MAE and RMSE reduction':f'{domain_both}/3',
    })
t1=pd.DataFrame(rows)
t1.to_csv(OUT/'Table1_target_level_summary.csv',index=False)

# SI machine-readable tables
pd.read_csv(RES/'stage_c'/'stage_c_skill_bootstrap_summary.csv').to_csv(OUT/'TableS1_stage_c_full_skill_bootstrap.csv',index=False)
pd.read_csv(RES/'stage_c'/'stage_c_threshold_sensitivity.csv').to_csv(OUT/'TableS2_threshold_sensitivity.csv',index=False)
pd.read_csv(RES/'stage_d'/'stage_d_domain_summary.csv').to_csv(OUT/'TableS3_domain_robustness_summary.csv',index=False)
pd.read_csv(RES/'stage_e'/'stage_e_estimator_earliest_stage_comparison.csv').to_csv(OUT/'TableS4_estimator_sensitivity.csv',index=False)

captions = '''# CMT figure captions - overlap-controlled draft\n\n## Figure 1\nTarget-specific information provenance and stage eligibility. (a) Candidate descriptors are assigned separately for each target using the expert-defined dependency graph and rule-based compiler. P1 contains framework-composition and working-ion descriptors. P2 adds target-eligible descriptors obtained from DFT-relaxed structures. P3 adds target-eligible post-DFT energetic, stability, and electronic information while direct target duplicates and target-defining components remain excluded from the clean representations. The compiler reproduced the curated reference classification for all 2,430 feature-target pairs. (b) Number of eligible descriptors for each target at P1, P2, and P3.\n\n## Figure 2\nStage-dependent learnability under framework-formula-grouped validation. DummyMean-relative MAE (a) and RMSE (b) skill are shown for the fixed ExtraTrees estimator at the composition (P1), relaxed-structure (P2), and legitimate post-DFT (P3) stages. Positive values indicate lower error than the fold-matched mean baseline. Black cell outlines mark the first stage satisfying the pre-specified useful-learnability rule: at least 10% improvement in both MAE and RMSE, with both 95% framework-group bootstrap intervals entirely above zero. Maximum volume change did not satisfy the full rule through P3.\n\n## Figure 3\nIncremental information value of later computational stages. Relative changes in MAE (a) and RMSE (b) are reported for P1 to P2, P2 to P3, and P1 to P3 under the fixed ExtraTrees framework-formula benchmark. Positive values indicate lower prediction error at the later stage. Points show observed relative improvement and horizontal bars show 95% paired framework-group bootstrap intervals. The dashed line marks zero change.\n\n## Figure 4\nChemical-domain robustness of the net P1-to-P3 information gain. Relative MAE (a) and RMSE (b) reductions are shown for framework-formula-grouped validation and for the host-chemical-system, coarse-chemistry-family, and working-ion holdouts. Positive values indicate lower error at P3 than at P1. Framework-formula values come from the primary analysis, whereas the three chemical-domain panels use descriptive macro point estimates. These holdout results are robustness checks and are not independent earliest-stage classifications.\n'''
(OUT/'figure_captions_overlap_controlled.md').write_text(captions,encoding='utf-8')

firewall = '''# Text-overlap firewall for the CMT manuscript\n\nThe CMT paper is framed around target-specific information provenance and stage-dependent learnability. Do not reuse the PLOS ONE manuscript's title logic, paragraph structure, or cross-database-transfer narrative.\n\n## Avoid as organizing language\n- internal validation is not transferability\n- cross-database transferability / transfer gap as the central claim\n- locked external database as the central evaluation design\n- leakage-aware cross-database assessment\n- Protocol A / Protocol B-strict language\n\n## Preferred CMT organizing language\n- target-specific descriptor eligibility\n- information provenance\n- composition, relaxed structure, and legitimate post-DFT stages\n- stage-dependent learnability\n- marginal information value\n- earliest useful information stage under the fixed primary estimator\n- chemical-domain robustness as supporting evidence\n\nAll figure captions in this publication-output package were written from the frozen CMT claims register rather than copied from the submitted PLOS ONE manuscript or the earlier electrode draft.\n'''
(OUT/'TEXT_OVERLAP_FIREWALL.md').write_text(firewall,encoding='utf-8')

# Plot-data snapshots
pd.DataFrame({
    'target':[labels[t] for t in order],
    **{f'MAE_skill_{s}':m[f'MAE_skill_{s}'] for s in stages},
    **{f'RMSE_skill_{s}':m[f'RMSE_skill_{s}'] for s in stages},
}).to_csv(OUT/'Fig2_plot_data.csv',index=False)

# Hash manifest
records=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='publication_outputs_sha256.csv':
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        records.append({'file':p.name,'sha256':h,'bytes':p.stat().st_size})
pd.DataFrame(records).to_csv(OUT/'publication_outputs_sha256.csv',index=False)

print(f'Publication outputs: PASS | figures=4 | main tables=1 | SI tables=4')
print(f'Output: {OUT}')
