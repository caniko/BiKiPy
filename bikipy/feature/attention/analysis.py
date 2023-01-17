import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


def attention_state_distribution(trials):
    attention_state_analysis = {
        "proximity&ray true observation false": [],
        "observation&ray true proximity false": [],
        "observation&proximity true ray false": [],
        "proximity true ray false": [],
        "ray true proximity false": [],
        "all false": [],
    }
    for trial in trials:
        attention_state_analysis["proximity&ray true observation false"].extend(
            (
                trial.a_proximity_filtered & trial.a_ray_filtered & ~trial.a_observance_per_frame,
                #
                trial.b_proximity_filtered & trial.b_ray_filtered & ~trial.b_observance_per_frame,
            )
        )
        attention_state_analysis["observation&ray true proximity false"].extend(
            (
                trial.a_observance_per_frame
                & trial.a_ray_filtered
                & (not_a_proximity_filtered := ~trial.a_proximity_filtered),
                #
                trial.b_observance_per_frame
                & trial.b_ray_filtered
                & (not_b_proximity_filtered := ~trial.b_proximity_filtered),
            ),
        )
        attention_state_analysis["observation&proximity true ray false"].extend(
            (
                trial.a_observance_per_frame
                & trial.a_proximity_filtered
                & (not_a_ray_filtered := ~trial.a_ray_filtered),
                #
                trial.b_observance_per_frame
                & trial.b_proximity_filtered
                & (not_b_ray_filtered := ~trial.b_ray_filtered),
            )
        )
        attention_state_analysis["proximity true ray false"].extend(
            (
                trial.a_proximity_filtered & not_a_ray_filtered,
                trial.b_proximity_filtered & not_b_ray_filtered,
            )
        )
        attention_state_analysis["ray true proximity false"].extend(
            (
                trial.a_ray_filtered & not_a_proximity_filtered,
                trial.b_ray_filtered & not_b_proximity_filtered,
            ),
        )

    result = []
    for label, data_set in attention_state_analysis.items():
        for idx, data in enumerate(data_set):
            analysis = np.sum(data) / data.size
            attention_state_analysis[label][idx] = analysis
            result.append((analysis, label))

    return pd.DataFrame(result, columns=("Ratio", "Comparison"))


def plot_attention_state_distribution(attention_state_distribution, bins=13, **sb_displot_kwargs):
    sb.set_theme(style="whitegrid")
    sb.displot(
        attention_state_distribution,
        x="Ratio",
        hue="Comparison",
        multiple="stack",
        bins=bins,
        **sb_displot_kwargs,
    )
    plt.show()
    sb.displot(
        attention_state_distribution,
        x="Ratio",
        hue="Comparison",
        multiple="stack",
        bins=bins,
        **sb_displot_kwargs,
    )
    plt.show()
