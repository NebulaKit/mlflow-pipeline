from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import warnings
from plotnine.exceptions import PlotnineWarning

from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# plotting
from plotnine import (
    ggplot, aes, geom_boxplot, geom_segment, geom_text, facet_wrap, scale_fill_brewer,
    labs, theme_bw, theme, element_text
)

# Silence plotnine filename/size warnings
warnings.filterwarnings("ignore", category=PlotnineWarning)


def plot_de_boxplots_for_top_features(
    top_features_per_class: Dict[str, List[str]],
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    group_col_name: str,
    control_name: str,
    output_dir: str,
    model_name: str,
    feature_map_path: Optional[str] = None,
    use_log2_fc: bool = True,
    tukey_alpha: float = 0.05,
    facet_cols: int = 5,
    palette_type: str = "qual",
    palette_name: str = "Pastel1",
    figure_size: tuple = (14, 9),
) -> Dict[str, str]:

    os.makedirs(output_dir, exist_ok=True)

    # 1) Establish the desired group order FROM THE DICT KEYS (your requested behavior)
    group_order = list(top_features_per_class.keys())
    if control_name not in group_order:
        raise ValueError(f"control_name='{control_name}' not found in dict keys {group_order}")

    # optional sanity check: make sure every y_raw value is in group_order
    unknown = sorted(set(y_raw.unique()) - set(group_order))
    if unknown:
        raise ValueError(f"Found labels in y_raw not present in dict keys: {unknown}")

    # Pretty labels mapping (unchanged)
    id_to_pretty = None
    if feature_map_path:
        meta = pd.read_csv(feature_map_path)
        assert {"ID", "Name"}.issubset(meta.columns), "feature_map must have columns ['ID','Name']"
        id_to_pretty = dict(zip(meta["ID"], meta["Name"]))

    # Working DF
    df = X_raw.copy()
    df[group_col_name] = y_raw.values

    class_plot_paths: Dict[str, str] = {}

    for class_name, feats in top_features_per_class.items():
        feats = [f for f in feats if f in df.columns]
        if not feats:
            continue

        # --- FC computation ---
        ctrl_means = df[df[group_col_name] == control_name][feats].mean()
        df_fc = df[[group_col_name] + feats].copy()
        if use_log2_fc:
            eps = 1e-12
            for f in feats:
                denom = ctrl_means.get(f, np.nan)
                df_fc[f] = df[f] if (pd.isna(denom) or denom == 0) else np.log2((df[f] + eps) / (denom + eps))
            y_label = "Fold Change (log2 vs Control)"
        else:
            for f in feats:
                denom = ctrl_means.get(f, np.nan)
                df_fc[f] = df[f] if (pd.isna(denom) or denom == 0) else (df[f] / denom)
            y_label = "Fold Change (vs Control)"

        # Melt
        df_top = df_fc.melt(id_vars=group_col_name, var_name="Feature", value_name="FoldChange")

        # 2) Force the plot x-axis to use the dict key order
        df_top[group_col_name] = pd.Categorical(df_top[group_col_name], categories=group_order, ordered=True)

        # facet labels (unchanged)
        if id_to_pretty:
            ordered_labels = [f"{id_to_pretty.get(x, x)} ({x})" for x in feats]
            df_top["LabelName"] = df_top["Feature"].map(lambda x: f"{id_to_pretty.get(x, x)} ({x})")
        else:
            ordered_labels = feats[:]
            df_top["LabelName"] = df_top["Feature"]
        df_top["LabelName"] = pd.Categorical(df_top["LabelName"], categories=ordered_labels, ordered=True)

        # --- ANOVA (use the same group order) ---
        anova_rows = []
        for f in feats:
            groups = [df[df[group_col_name] == g][f].dropna().values for g in group_order]
            if sum(len(gv) > 0 for gv in groups) >= 2:
                stat, pval = f_oneway(*groups)
            else:
                stat, pval = np.nan, np.nan

            # Add pretty label if available
            label_name = f"{id_to_pretty.get(f, f)} ({f})" if id_to_pretty else f
            anova_rows.append({
                "Feature": f,
                "LabelName": label_name,
                "F_stat": stat,
                "p_value": pval
            })

        anova_df = pd.DataFrame(anova_rows)
        anova_df["adj_p_value"] = multipletests(anova_df["p_value"], method="fdr_bh")[1]
        anova_df.to_csv(os.path.join(output_dir, f"{class_name}_anova.csv"), index=False)

        # --- Tukey (also force the same group order) ---
        signif_rows = []
        for f in feats:
            sub = df[[group_col_name, f]].copy().dropna().rename(columns={f: "Concentration"})
            sub[group_col_name] = pd.Categorical(sub[group_col_name], categories=group_order, ordered=True)
            if sub[group_col_name].nunique() >= 2:
                try:
                    tukey = pairwise_tukeyhsd(sub["Concentration"], sub[group_col_name], alpha=tukey_alpha)
                    for r in tukey.summary().data[1:]:
                        g1, g2, _, p_adj, _, _, reject = r
                        if reject:
                            stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*"
                            signif_rows.append({"Feature": f, "Group1": g1, "Group2": g2, "label": stars})
                except Exception:
                    pass

        sig_df = pd.DataFrame(signif_rows)
        if not sig_df.empty:
            if id_to_pretty:
                sig_df["LabelName"] = sig_df["Feature"].map(lambda x: f"{id_to_pretty.get(x, x)} ({x})")
            else:
                sig_df["LabelName"] = sig_df["Feature"]
            sig_df["LabelName"] = pd.Categorical(sig_df["LabelName"], categories=ordered_labels, ordered=True)

            # 3) Map x positions using the SAME dict-order map
            group_pos = {g: i + 1 for i, g in enumerate(group_order)}
            sig_df["x1"] = sig_df["Group1"].map(group_pos)
            sig_df["x2"] = sig_df["Group2"].map(group_pos)
            sig_df["xmid"] = (sig_df["x1"] + sig_df["x2"]) / 2

            base_y = (
                df_top.groupby("LabelName", observed=False)["FoldChange"]
                .max()
                .reset_index()
                .rename(columns={"FoldChange": "base_y"})
            )
            sig_df = sig_df.merge(base_y, on="LabelName", how="left")
            sig_df["line_index"] = sig_df.groupby("LabelName", observed=False).cumcount()
            sig_df["ystart"] = sig_df["base_y"] + (sig_df["line_index"] + 1) * 0.1
            sig_df["ytext"] = sig_df["ystart"] + 0.02

            sig_df.to_csv(os.path.join(output_dir, f"{class_name}_tukey.csv"), index=False)

        # Plot
        p = (
            ggplot(df_top, aes(x=group_col_name, y="FoldChange", fill=group_col_name))
            + geom_boxplot(outlier_size=0.5, alpha=0.85, width=0.65, show_legend=False)
            + (geom_segment(
                    data=sig_df,
                    mapping=aes(x="x1", xend="x2", y="ystart", yend="ystart"),
                    size=0.3, color="black", inherit_aes=False
               ) if not sig_df.empty else None)
            + (geom_text(
                    data=sig_df,
                    mapping=aes(x="xmid", y="ytext", label="label"),
                    size=10, inherit_aes=False
               ) if not sig_df.empty else None)
            + facet_wrap("~LabelName", scales="free_y", ncol=facet_cols)
            + scale_fill_brewer(type=palette_type, palette=palette_name)
            + labs(
                title=f"Top {model_name} SHAP Features for Class {class_name}: "
                      f"Concentration Fold Change vs Control ({control_name})",
                x="Group",
                y=y_label
            )
            + theme_bw()
            + theme(
                figure_size=figure_size,
                axis_text_x=element_text(rotation=45, ha="right"),
                strip_text=element_text(weight="bold"),
                legend_position="none",
            )
        )

        out_path = os.path.join(output_dir, f"{class_name}_DE.png")
        p.save(out_path, dpi=150)
        class_plot_paths[class_name] = out_path

    return class_plot_paths
