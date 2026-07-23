
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv
import hashlib
import json
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


MAIN_DIR = Path("figures/main")
SUPP_DIR = Path("figures/supplementary")
SOURCE_DIR = Path("figures/source_data")
DPI = 600

TARGET_ORDER = [
    "average_voltage",
    "capacity_grav",
    "energy_grav",
    "max_delta_volume",
    "stability_worst",
]
TARGET_LABELS = {
    "average_voltage": "Voltage",
    "capacity_grav": "Gravimetric capacity",
    "capacity_vol": "Volumetric capacity",
    "energy_grav": "Gravimetric energy",
    "energy_vol": "Volumetric energy",
    "max_delta_volume": "Volume change",
    "stability_worst": "Worst stability",
    "stability_charge": "Charged-state stability",
    "stability_discharge": "Discharged-state stability",
}
SPLIT_ORDER = [
    "random_split",
    "framework_groupkfold",
    "leave_chemical_system_out",
    "leave_family_out",
    "leave_working_ion_out",
]
SPLIT_LABELS = {
    "random_split": "Random",
    "framework_groupkfold": "Framework",
    "leave_chemical_system_out": "Chemical system",
    "leave_family_out": "Family",
    "leave_working_ion_out": "Working ion",
}
PROTOCOL_LABELS = {
    "P0": "P0 full-feature baseline",
    "P1": "P1 composition-only",
    "P2": "P2 + relaxed structure",
    "P3": "P3 post-DFT clean",
    "P4": "P4 direct-target stress test",
}

INPUT_FILES = [
    "results/audits/notebook_01/01_record_counts_by_ion.csv",
    "results/audits/notebook_01/01_family_counts_by_ion.csv",
    "results/audits/notebook_02/02_protocol_feature_counts_by_target.csv",
    "results/audits/notebook_02/02_target_specific_leakage_summary.csv",
    "results/audits/notebook_03/03_compiler_validation_summary.csv",
    "data/processed/notebook_04/04_compact_best_model_benchmark_table.csv",
    "results/metrics/notebook_05/05_independent_vs_hardderived_vs_softconstrained_metrics.csv",
    "results/metrics/notebook_05/05_physics_consistency_metrics.csv",
    "data/processed/notebook_07/07_error_by_applicability_domain_flag.csv",
    "data/processed/notebook_07/07_compact_uncertainty_ad_table.csv",
    "data/processed/notebook_08/08_sodium_case_study_candidate_table_compact.csv",
    "results/audits/notebook_08/08_sodium_screening_funnel.csv",
    "data/processed/notebook_10/10_dft_primary_selection.csv",
    "dft_validation/results/kpoint_convergence_trend.tsv",
    "dft_validation/results/selected_vs_reference_voltage.tsv",
]


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "data" / "processed" / "notebook_01").is_dir()
            and (candidate / "results").is_dir()
            and (candidate / "dft_validation").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Repository root not found. Run from the repository or notebooks directory."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / f"{stem}.pdf",
        directory / f"{stem}.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_source(df: pd.DataFrame, stem: str) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    stem: str,
    annotation_format: str = ".2f",
    log10: bool = False,
    directory: Path = MAIN_DIR,
) -> list[Path]:
    values = np.log10(matrix.to_numpy(dtype=float)) if log10 else matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(
        np.arange(matrix.shape[1]),
        [SPLIT_LABELS.get(value, value) for value in matrix.columns],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        np.arange(matrix.shape[0]),
        [TARGET_LABELS.get(value, value) for value in matrix.index],
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)

    finite = values[np.isfinite(values)]
    threshold = np.nanmedian(finite) if finite.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            original_value = matrix.iloc[row_index, column_index]
            plotted_value = values[row_index, column_index]
            if pd.isna(original_value):
                label = "NA"
            else:
                label = format(original_value, annotation_format)
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if plotted_value > threshold else "black",
            )

    ax.set_title(title)
    return save_figure(fig, directory, stem)


def figure_1_workflow(root: Path) -> list[Path]:
    counts = pd.read_csv(
        root / "results/audits/notebook_01/01_record_counts_by_ion.csv"
    ).set_index("working_ion")["n_records"]
    voltage = pd.read_csv(
        root / "dft_validation/results/selected_vs_reference_voltage.tsv",
        sep="\t",
    )
    selected = voltage.loc[voltage["set"] == "selected_candidate"].iloc[0]
    reference = voltage.loc[voltage["set"] == "denser_reference"].iloc[0]

    steps = [
        (
            "1. Multi-ion data extraction",
            f"Li {int(counts['Li']):,} | Na {int(counts['Na']):,} | K {int(counts['K']):,}",
        ),
        (
            "2. Battery-property dependency graph",
            "Machine-readable target–feature relationships",
        ),
        (
            "3. Automatic leakage compiler",
            "Target-specific P0–P4 descriptor protocols",
        ),
        (
            "4. Leakage-aware validation",
            "Random, framework, family, chemical-system and ion holdouts",
        ),
        (
            "5. Physics-constrained multi-task learning",
            "Voltage, capacity, energy, volume and stability",
        ),
        (
            "6. Uncertainty and applicability domain",
            "Conformal intervals, tree spread and distance flags",
        ),
        (
            "7. Sodium candidate triage",
            "Physics filters, transfer score and rank robustness",
        ),
        (
            "8. Quantum ESPRESSO spot-check",
            (
                f"Fixed-geometry PBE {selected['average_voltage_V']:.3f} V; "
                f"|ΔV| to denser reference "
                f"{reference['abs_delta_voltage_from_selected_V']:.5f} V"
            ),
        ),
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_positions = np.linspace(0.92, 0.08, len(steps))

    for index, ((title, subtitle), y_position) in enumerate(
        zip(steps, y_positions)
    ):
        x_position = 0.10 if index % 2 == 0 else 0.18
        width = 0.72
        color = colors[index % len(colors)]
        box = FancyBboxPatch(
            (x_position, y_position - 0.045),
            width,
            0.085,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=color,
            alpha=0.16,
            edgecolor=color,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(
            x_position + 0.02,
            y_position + 0.012,
            title,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            x_position + 0.02,
            y_position - 0.021,
            subtitle,
            va="center",
            ha="left",
            fontsize=9,
        )
        if index < len(steps) - 1:
            next_x = 0.10 if (index + 1) % 2 == 0 else 0.18
            ax.add_patch(
                FancyArrowPatch(
                    (x_position + width / 2, y_position - 0.048),
                    (
                        next_x + width / 2,
                        y_positions[index + 1] + 0.048,
                    ),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    connectionstyle="arc3,rad=0.08",
                )
            )

    ax.text(
        0.5,
        0.985,
        "Leakage-audited multi-ion insertion-electrode learning workflow",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(fig, MAIN_DIR, "Figure_1_workflow")


def figure_2_dataset_landscape(root: Path) -> list[Path]:
    family_counts = pd.read_csv(
        root / "results/audits/notebook_01/01_family_counts_by_ion.csv"
    )
    pivot = family_counts.pivot(
        index="working_ion",
        columns="coarse_family",
        values="n_records",
    ).fillna(0)
    ion_order = ["Li", "Na", "K"]
    pivot = pivot.reindex(ion_order)
    percentages = pivot.div(pivot.sum(axis=1), axis=0) * 100
    top_families = pivot.sum().sort_values(ascending=False).head(7).index
    compact = percentages[top_families].copy()
    compact["Other families"] = 100 - compact.sum(axis=1)

    human_names = {
        column: column.replace("_", " ").replace(" like", "-like")
        for column in compact.columns
    }
    human_names["polyanion_phosphate_like"] = "Polyanion/phosphate-like"
    human_names["transition_metal_oxide_like"] = "Transition-metal oxide-like"
    human_names["halide_or_fluoride_like"] = "Halide/fluoride-like"
    human_names["oxyfluoride_or_fluoropolyanion_like"] = (
        "Oxyfluoride/fluoropolyanion-like"
    )
    human_names["sulfate_or_sulfur_oxoanion_like"] = (
        "Sulfate/sulfur-oxoanion-like"
    )
    human_names["other_or_unclear"] = "Other/unclear"

    source = compact.reset_index().rename(
        columns={column: human_names[column] for column in compact.columns}
    )
    counts = pivot.sum(axis=1).astype(int)
    source.insert(
        1,
        "n_records",
        source["working_ion"].map(counts.to_dict()),
    )
    save_source(source, "Figure_2_dataset_landscape_source")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bottom = np.zeros(len(compact))
    x_positions = np.arange(len(compact))

    for column in compact.columns:
        values = compact[column].to_numpy()
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=human_names[column],
        )
        bottom += values

    tick_labels = [
        f"{ion}\n(n={int(counts.loc[ion]):,})"
        for ion in ion_order
    ]
    ax.set_xticks(x_positions, tick_labels)
    ax.set_ylabel("Share of records (%)")
    ax.set_xlabel("Working ion")
    ax.set_ylim(0, 100)
    ax.legend(
        title="Coarse chemical family",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    ax.set_title("Chemical-family composition of the multi-ion dataset")
    return save_figure(fig, MAIN_DIR, "Figure_2_dataset_landscape")


def figure_3_leakage_validation(root: Path) -> list[Path]:
    benchmark = pd.read_csv(
        root / "data/processed/notebook_04/"
        "04_compact_best_model_benchmark_table.csv"
    )
    paired = (
        benchmark.loc[benchmark["protocol"].isin(["P0", "P1"])]
        .pivot_table(
            index=["target", "split_name"],
            columns="protocol",
            values="mae_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    paired["clean_to_leaky_mae_ratio"] = paired["P1"] / paired["P0"]
    source = paired[
        ["target", "split_name", "P0", "P1", "clean_to_leaky_mae_ratio"]
    ].copy()
    source = source.rename(
        columns={
            "P0": "P0_leaky_baseline_MAE",
            "P1": "P1_clean_MAE",
        }
    )
    save_source(source, "Figure_3_leakage_validation_source")

    matrix = (
        paired.pivot(
            index="target",
            columns="split_name",
            values="clean_to_leaky_mae_ratio",
        )
        .reindex(index=TARGET_ORDER, columns=SPLIT_ORDER)
    )
    display = matrix.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: value if pd.isna(value) else value
        )

    values = np.log10(matrix.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(
        np.arange(len(SPLIT_ORDER)),
        [SPLIT_LABELS[value] for value in SPLIT_ORDER],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(TARGET_ORDER)),
        [TARGET_LABELS[value] for value in TARGET_ORDER],
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        r"$\log_{10}$(clean P1 MAE / leaky P0 MAE)"
    )
    threshold = np.nanmedian(values)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ratio = matrix.iloc[row_index, column_index]
            plotted = values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{ratio:.1f}×",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if plotted > threshold else "black",
            )
    ax.set_title(
        "Performance optimism from post-hoc leakage across validation regimes"
    )
    return save_figure(fig, MAIN_DIR, "Figure_3_leakage_validation")


def figure_4_physics_constraints(root: Path) -> list[Path]:
    metrics = pd.read_csv(
        root / "results/metrics/notebook_05/"
        "05_physics_consistency_metrics.csv"
    )
    grouped = (
        metrics.loc[
            (metrics["protocol"] == "P1")
            & metrics["model_variant"].isin(
                ["unconstrained_direct", "softconstrained_direct"]
            )
        ]
        .groupby(["split_name", "model_variant"])
        .agg(
            energy_consistency_mae=("energy_consistency_mae", "mean"),
            stability_consistency_mae=("stability_consistency_mae", "mean"),
        )
        .reset_index()
    )
    pivot = grouped.pivot(
        index="split_name",
        columns="model_variant",
        values=["energy_consistency_mae", "stability_consistency_mae"],
    )

    rows: list[dict[str, float | str]] = []
    for split_name in SPLIT_ORDER:
        unconstrained_energy = pivot.loc[
            split_name,
            ("energy_consistency_mae", "unconstrained_direct"),
        ]
        constrained_energy = pivot.loc[
            split_name,
            ("energy_consistency_mae", "softconstrained_direct"),
        ]
        unconstrained_stability = pivot.loc[
            split_name,
            ("stability_consistency_mae", "unconstrained_direct"),
        ]
        constrained_stability = pivot.loc[
            split_name,
            ("stability_consistency_mae", "softconstrained_direct"),
        ]
        rows.append({
            "split_name": split_name,
            "unconstrained_energy_consistency_mae": unconstrained_energy,
            "softconstrained_energy_consistency_mae": constrained_energy,
            "energy_consistency_reduction_pct": (
                100
                * (unconstrained_energy - constrained_energy)
                / unconstrained_energy
            ),
            "unconstrained_stability_consistency_mae": unconstrained_stability,
            "softconstrained_stability_consistency_mae": constrained_stability,
            "stability_consistency_reduction_pct": (
                100
                * (unconstrained_stability - constrained_stability)
                / unconstrained_stability
            ),
        })

    source = pd.DataFrame(rows)
    save_source(source, "Figure_4_physics_constraints_source")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x_positions = np.arange(len(source))
    width = 0.36
    energy = source["energy_consistency_reduction_pct"].to_numpy()
    stability = source["stability_consistency_reduction_pct"].to_numpy()
    ax.bar(
        x_positions - width / 2,
        energy,
        width,
        label="Energy consistency",
    )
    ax.bar(
        x_positions + width / 2,
        stability,
        width,
        label="Stability consistency",
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(
        x_positions,
        [SPLIT_LABELS[value] for value in source["split_name"]],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Reduction in physical inconsistency (%)")
    ax.set_title(
        "Value of soft physics constraints under the clean composition-only protocol"
    )
    ax.legend(frameon=False)
    for x_position, value in zip(x_positions - width / 2, energy):
        ax.text(
            x_position,
            value + 1,
            f"{value:.0f}%",
            ha="center",
            fontsize=8,
        )
    for x_position, value in zip(x_positions + width / 2, stability):
        ax.text(
            x_position,
            value + 1,
            f"{value:.0f}%",
            ha="center",
            fontsize=8,
        )
    return save_figure(fig, MAIN_DIR, "Figure_4_physics_constraints")


def figure_5_applicability_domain(root: Path) -> list[Path]:
    errors = pd.read_csv(
        root / "data/processed/notebook_07/"
        "07_error_by_applicability_domain_flag.csv"
    )
    paired = (
        errors.loc[errors["protocol"] == "P1"]
        .pivot_table(
            index=["target", "split_name"],
            columns="ad_flag",
            values="mae",
            aggfunc="first",
        )
        .reset_index()
    )
    paired["out_to_in_domain_mae_ratio"] = (
        paired["out_of_domain"] / paired["in_domain"]
    )
    save_source(
        paired[
            [
                "target",
                "split_name",
                "in_domain",
                "borderline",
                "out_of_domain",
                "out_to_in_domain_mae_ratio",
            ]
        ],
        "Figure_5_applicability_domain_source",
    )

    matrix = (
        paired.pivot(
            index="target",
            columns="split_name",
            values="out_to_in_domain_mae_ratio",
        )
        .reindex(index=TARGET_ORDER, columns=SPLIT_ORDER)
    )
    return heatmap(
        matrix,
        (
            "Applicability-domain flags identify elevated error "
            "for several targets"
        ),
        "Out-of-domain MAE / in-domain MAE",
        "Figure_5_applicability_domain",
        annotation_format=".2f",
        log10=False,
        directory=MAIN_DIR,
    )


def figure_6_sodium_dft(root: Path) -> list[Path]:
    candidates = pd.read_csv(
        root / "data/processed/notebook_08/"
        "08_sodium_case_study_candidate_table_compact.csv"
    )
    primary = pd.read_csv(
        root / "data/processed/notebook_10/10_dft_primary_selection.csv"
    )
    voltage = pd.read_csv(
        root / "dft_validation/results/selected_vs_reference_voltage.tsv",
        sep="\t",
    )
    selected_voltage = voltage.loc[
        voltage["set"] == "selected_candidate"
    ].iloc[0]
    reference_voltage = voltage.loc[
        voltage["set"] == "denser_reference"
    ].iloc[0]

    selected_candidates = primary.loc[
        primary["selection_role"].astype(str).str.contains(
            "primary", case=False, na=False
        )
    ].copy()
    exact_formula = "Na1-3CoPCO7"
    selected = selected_candidates.loc[
        selected_candidates["battery_formula"] == exact_formula
    ]
    if selected.empty:
        selected = selected_candidates.iloc[[0]]
    selected = selected.iloc[0]

    screen = candidates.loc[candidates["case_screen_pass"]].copy()
    source_columns = [
        "shortlist_priority_rank",
        "battery_formula",
        "formula_charge",
        "formula_discharge",
        "average_voltage",
        "capacity_grav",
        "energy_grav",
        "max_delta_volume",
        "stability_worst",
        "final_triage_score",
        "rank_median_mc",
        "rank_iqr_mc",
        "out_of_domain_flag",
        "high_uncertainty_flag",
    ]
    save_source(
        screen[source_columns],
        "Figure_6_sodium_candidates_source",
    )
    save_source(
        voltage,
        "Figure_6_dft_voltage_source",
    )

    score_min = screen["final_triage_score"].min()
    score_max = screen["final_triage_score"].max()
    denominator = max(score_max - score_min, 1e-12)
    sizes = (
        30
        + 260
        * (screen["final_triage_score"] - score_min)
        / denominator
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        screen["stability_worst"],
        screen["energy_grav"],
        s=sizes,
        alpha=0.45,
        label="Screen-passing Na candidates",
    )
    ax.scatter(
        [selected["stability_worst"]],
        [selected["energy_grav"]],
        s=180,
        marker="*",
        linewidths=1.4,
        label="Selected DFT case",
    )
    ax.annotate(
        r"Na$_{1-3}$CoPCO$_7$",
        (selected["stability_worst"], selected["energy_grav"]),
        xytext=(18, -18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlabel(r"Worst endpoint stability (eV atom$^{-1}$)")
    ax.set_ylabel(r"Gravimetric energy (Wh kg$^{-1}$)")
    ax.set_title(
        "Sodium candidate triage and original fixed-geometry DFT spot-check"
    )
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.98,
        0.97,
        (
            "Fixed-geometry PBE\n"
            f"selected mesh: {selected_voltage['average_voltage_V']:.6f} V\n"
            f"dense reference: {reference_voltage['average_voltage_V']:.6f} V\n"
            f"|ΔV| = "
            f"{reference_voltage['abs_delta_voltage_from_selected_V']:.6f} V"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )
    return save_figure(fig, MAIN_DIR, "Figure_6_sodium_dft")


def supplementary_1_protocol_feature_counts(root: Path) -> list[Path]:
    counts = pd.read_csv(
        root / "results/audits/notebook_02/"
        "02_protocol_feature_counts_by_target.csv"
    )
    source = counts[
        ["target", "protocol", "protocol_name", "n_features"]
    ].copy()
    save_source(source, "Figure_S1_protocol_feature_counts_source")
    matrix = (
        source.pivot(
            index="target",
            columns="protocol",
            values="n_features",
        )
        .reindex(index=[
            "average_voltage",
            "capacity_grav",
            "capacity_vol",
            "energy_grav",
            "energy_vol",
            "max_delta_volume",
            "stability_charge",
            "stability_discharge",
            "stability_worst",
        ], columns=["P0", "P1", "P2", "P3", "P4"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(
        np.arange(matrix.shape[1]),
        [PROTOCOL_LABELS[value] for value in matrix.columns],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(
        np.arange(matrix.shape[0]),
        [TARGET_LABELS.get(value, value.replace("_", " ")) for value in matrix.index],
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Number of input features")
    threshold = np.nanmedian(matrix.to_numpy(dtype=float))
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    ax.set_title("Target-specific feature counts across leakage protocols")
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S1_protocol_feature_counts",
    )


def supplementary_2_compiler_validation(root: Path) -> list[Path]:
    validation = pd.read_csv(
        root / "results/audits/notebook_03/"
        "03_compiler_validation_summary.csv"
    )
    validation = validation.loc[validation["target"] != "ALL"].copy()
    save_source(
        validation,
        "Figure_S2_compiler_validation_source",
    )

    labels = [
        TARGET_LABELS.get(value, value.replace("_", " "))
        for value in validation["target"]
    ]
    y_positions = np.arange(len(validation))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        validation["class_agreement_rate"] * 100,
        y_positions,
        label="Leakage-class agreement",
    )
    ax.scatter(
        validation["all_protocols_agree_rate"] * 100,
        y_positions,
        marker="x",
        label="All-protocol agreement",
    )
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(95, 100.25)
    ax.set_xlabel("Agreement with expert audit (%)")
    ax.set_title(
        "Automatic leakage compiler reproduced the expert audit"
    )
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        (
            f"Total audited target–feature rows: "
            f"{int(validation['n_rows'].sum()):,}; "
            f"false-safe classifications: "
            f"{int(validation['false_safe_n'].sum())}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S2_compiler_validation",
    )


def supplementary_3_clean_domain_shift(root: Path) -> list[Path]:
    benchmark = pd.read_csv(
        root / "data/processed/notebook_04/"
        "04_compact_best_model_benchmark_table.csv"
    )
    clean = benchmark.loc[benchmark["protocol"] == "P1"].copy()
    random_mae = (
        clean.loc[clean["split_name"] == "random_split", ["target", "mae_mean"]]
        .rename(columns={"mae_mean": "random_mae"})
    )
    clean = clean.merge(random_mae, on="target", how="left")
    clean["mae_ratio_vs_random"] = clean["mae_mean"] / clean["random_mae"]
    save_source(
        clean[
            [
                "target",
                "split_name",
                "model_name",
                "mae_mean",
                "random_mae",
                "mae_ratio_vs_random",
            ]
        ],
        "Figure_S3_clean_domain_shift_source",
    )
    matrix = (
        clean.pivot(
            index="target",
            columns="split_name",
            values="mae_ratio_vs_random",
        )
        .reindex(index=TARGET_ORDER, columns=SPLIT_ORDER)
    )
    return heatmap(
        matrix,
        "Clean-protocol MAE relative to random splitting",
        "P1 MAE / random-split P1 MAE",
        "Figure_S3_clean_domain_shift",
        annotation_format=".2f",
        directory=SUPP_DIR,
    )


def supplementary_4_predictive_tradeoff(root: Path) -> list[Path]:
    metrics = pd.read_csv(
        root / "results/metrics/notebook_05/"
        "05_independent_vs_hardderived_vs_softconstrained_metrics.csv"
    )
    direct = metrics.loc[
        (metrics["protocol"] == "P1")
        & metrics["model_variant"].isin(
            ["unconstrained_direct", "softconstrained_direct"]
        )
    ].copy()
    paired = (
        direct.pivot_table(
            index=["target", "split_name"],
            columns="model_variant",
            values="mae_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    paired["soft_to_unconstrained_mae_ratio"] = (
        paired["softconstrained_direct"]
        / paired["unconstrained_direct"]
    )
    save_source(
        paired,
        "Figure_S4_predictive_tradeoff_source",
    )
    target_order = [
        "average_voltage",
        "capacity_grav",
        "energy_grav",
        "max_delta_volume",
        "stability_charge",
        "stability_discharge",
        "stability_worst",
    ]
    matrix = (
        paired.pivot(
            index="target",
            columns="split_name",
            values="soft_to_unconstrained_mae_ratio",
        )
        .reindex(index=target_order, columns=SPLIT_ORDER)
    )
    return heatmap(
        matrix,
        "Predictive MAE trade-off of soft physics constraints",
        "Soft-constrained MAE / unconstrained MAE",
        "Figure_S4_predictive_tradeoff",
        annotation_format=".3f",
        directory=SUPP_DIR,
    )


def supplementary_5_conformal_coverage(root: Path) -> list[Path]:
    uncertainty = pd.read_csv(
        root / "data/processed/notebook_07/"
        "07_compact_uncertainty_ad_table.csv"
    )
    clean = uncertainty.loc[uncertainty["protocol"] == "P1"].copy()
    save_source(
        clean[
            [
                "target",
                "split_name",
                "n_predictions",
                "conformal_coverage",
                "median_conformal_width",
            ]
        ],
        "Figure_S5_conformal_coverage_source",
    )
    matrix = (
        clean.pivot(
            index="target",
            columns="split_name",
            values="conformal_coverage",
        )
        .reindex(index=TARGET_ORDER, columns=SPLIT_ORDER)
    )
    return heatmap(
        matrix,
        "Conformal interval coverage under the clean protocol",
        "Observed conformal coverage",
        "Figure_S5_conformal_coverage",
        annotation_format=".3f",
        directory=SUPP_DIR,
    )


def supplementary_6_out_of_domain_fraction(root: Path) -> list[Path]:
    uncertainty = pd.read_csv(
        root / "data/processed/notebook_07/"
        "07_compact_uncertainty_ad_table.csv"
    )
    clean = uncertainty.loc[uncertainty["protocol"] == "P1"].copy()
    clean["in_domain_pct"] = clean["pct_in_domain"] * 100
    clean["borderline_pct"] = clean["pct_borderline"] * 100
    clean["out_of_domain_pct"] = clean["pct_out_of_domain"] * 100
    save_source(
        clean[
            [
                "target",
                "split_name",
                "in_domain_pct",
                "borderline_pct",
                "out_of_domain_pct",
            ]
        ],
        "Figure_S6_out_of_domain_fraction_source",
    )
    matrix = (
        clean.pivot(
            index="target",
            columns="split_name",
            values="out_of_domain_pct",
        )
        .reindex(index=TARGET_ORDER, columns=SPLIT_ORDER)
    )
    return heatmap(
        matrix,
        "Percentage of predictions flagged out of domain",
        "Out-of-domain predictions (%)",
        "Figure_S6_out_of_domain_fraction",
        annotation_format=".1f",
        directory=SUPP_DIR,
    )


def supplementary_7_sodium_funnel(root: Path) -> list[Path]:
    funnel = pd.read_csv(
        root / "results/audits/notebook_08/08_sodium_screening_funnel.csv"
    )
    save_source(
        funnel,
        "Figure_S7_sodium_screening_funnel_source",
    )
    display = funnel.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y_positions = np.arange(len(display))
    ax.barh(y_positions, display["n_records"])
    ax.set_yticks(
        y_positions,
        [textwrap.fill(value, 34) for value in display["stage"]],
    )
    ax.set_xlabel("Number of sodium insertion-electrode records")
    ax.set_title("Sodium case-study screening criteria")
    for y_position, value in zip(y_positions, display["n_records"]):
        ax.text(
            value + max(display["n_records"]) * 0.01,
            y_position,
            f"{int(value)}",
            va="center",
            fontsize=8,
        )
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S7_sodium_screening_funnel",
    )


def supplementary_8_rank_robustness(root: Path) -> list[Path]:
    candidates = pd.read_csv(
        root / "data/processed/notebook_08/"
        "08_sodium_case_study_candidate_table_compact.csv"
    ).sort_values("shortlist_priority_rank")
    top = candidates.head(20).copy()
    save_source(
        top[
            [
                "shortlist_priority_rank",
                "battery_formula",
                "rank_median_mc",
                "rank_iqr_mc",
                "top10_probability_mc",
                "top20_probability_mc",
                "final_triage_score",
            ]
        ],
        "Figure_S8_rank_robustness_source",
    )
    x_positions = top["shortlist_priority_rank"].to_numpy()
    y_values = top["rank_median_mc"].to_numpy()
    y_errors = top["rank_iqr_mc"].to_numpy() / 2
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        x_positions,
        y_values,
        yerr=y_errors,
        fmt="o",
        capsize=3,
        label="Monte Carlo median rank ± half-IQR",
    )
    ax.plot([1, 20], [1, 20], linestyle="--", label="Identity")
    ax.set_xlabel("Deterministic shortlist rank")
    ax.set_ylabel("Monte Carlo median rank")
    ax.set_xticks(np.arange(1, 21, 2))
    ax.set_title("Rank robustness of the top sodium candidates")
    ax.legend(frameon=False)
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S8_rank_robustness",
    )


def supplementary_9_kpoint_convergence(root: Path) -> list[Path]:
    trend = pd.read_csv(
        root / "dft_validation/results/kpoint_convergence_trend.tsv",
        sep="\t",
    )
    save_source(
        trend,
        "Figure_S9_kpoint_convergence_source",
    )
    labels = [
        f"{row.lower_mesh} → {row.upper_mesh}"
        for row in trend.itertuples()
    ]
    values = trend["abs_energy_delta_meV_atom"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_positions = np.arange(len(values))
    ax.plot(x_positions, values, marker="o")
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel(r"|ΔE| (meV atom$^{-1}$)")
    ax.set_xlabel("Charged-state k-point refinement")
    ax.set_title("Charged-state k-point energy convergence")
    for x_position, value in zip(x_positions, values):
        ax.text(
            x_position,
            value + max(values) * 0.05,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, max(values) * 1.25)
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S9_kpoint_convergence",
    )


def supplementary_10_voltage_convergence(root: Path) -> list[Path]:
    voltage = pd.read_csv(
        root / "dft_validation/results/selected_vs_reference_voltage.tsv",
        sep="\t",
    )
    save_source(
        voltage,
        "Figure_S10_voltage_convergence_source",
    )
    values = voltage["average_voltage_V"].to_numpy()
    labels = ["Selected meshes", "Denser reference"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x_positions = np.arange(len(values))
    ax.plot(x_positions, values, marker="o", linewidth=1.2)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Average voltage (V)")
    ax.set_title("Fixed-geometry voltage convergence")
    lower = min(values) - 0.00025
    upper = max(values) + 0.00025
    ax.set_ylim(lower, upper)
    for x_position, value in zip(x_positions, values):
        ax.text(
            x_position,
            value + 0.00004,
            f"{value:.6f} V",
            ha="center",
            fontsize=9,
        )
    difference = voltage.loc[
        voltage["set"] == "denser_reference",
        "abs_delta_voltage_from_selected_V",
    ].iloc[0]
    ax.text(
        0.5,
        0.08,
        f"|ΔV| = {difference:.6f} V",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    return save_figure(
        fig,
        SUPP_DIR,
        "Figure_S10_voltage_convergence",
    )


def build_figure_manifest(root: Path, paths: Iterable[Path]) -> Path:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = root / "figures/figure_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def build_source_hash_manifest(root: Path) -> Path:
    rows = []
    for relative in INPUT_FILES:
        path = root / relative
        rows.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    path = root / "figures/source_data/source_file_hashes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def generate_all_figures(root: Path | None = None) -> dict[str, object]:
    root = repository_root(root)
    style()

    global MAIN_DIR, SUPP_DIR, SOURCE_DIR
    MAIN_DIR = root / "figures/main"
    SUPP_DIR = root / "figures/supplementary"
    SOURCE_DIR = root / "figures/source_data"

    for directory in [MAIN_DIR, SUPP_DIR, SOURCE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.extend(figure_1_workflow(root))
    generated.extend(figure_2_dataset_landscape(root))
    generated.extend(figure_3_leakage_validation(root))
    generated.extend(figure_4_physics_constraints(root))
    generated.extend(figure_5_applicability_domain(root))
    generated.extend(figure_6_sodium_dft(root))

    generated.extend(supplementary_1_protocol_feature_counts(root))
    generated.extend(supplementary_2_compiler_validation(root))
    generated.extend(supplementary_3_clean_domain_shift(root))
    generated.extend(supplementary_4_predictive_tradeoff(root))
    generated.extend(supplementary_5_conformal_coverage(root))
    generated.extend(supplementary_6_out_of_domain_fraction(root))
    generated.extend(supplementary_7_sodium_funnel(root))
    generated.extend(supplementary_8_rank_robustness(root))
    generated.extend(supplementary_9_kpoint_convergence(root))
    generated.extend(supplementary_10_voltage_convergence(root))

    source_hashes = build_source_hash_manifest(root)
    all_figure_files = [
        path
        for directory in [MAIN_DIR, SUPP_DIR]
        for path in directory.iterdir()
        if path.is_file()
    ]
    manifest = build_figure_manifest(
        root,
        [*all_figure_files, *SOURCE_DIR.glob("*.csv")],
    )

    result = {
        "schema_version": "1.1",
        "repository_root": ".",
        "main_figure_files": len(list(MAIN_DIR.iterdir())),
        "supplementary_figure_files": len(list(SUPP_DIR.iterdir())),
        "source_data_files": len(list(SOURCE_DIR.iterdir())),
        "figure_manifest": manifest.relative_to(root).as_posix(),
        "source_hash_manifest": source_hashes.relative_to(root).as_posix(),
    }
    receipt = root / "figures/figure_generation_receipt.json"
    receipt.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(generate_all_figures(), indent=2))
