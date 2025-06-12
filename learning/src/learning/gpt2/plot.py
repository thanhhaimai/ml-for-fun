from typing import Self

import pandas as pd
import plotly.graph_objects as go

from learning.gpt2.metrics import ProbsMetrics


class Plotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def from_metrics(cls, metrics: list[list[ProbsMetrics]]) -> Self:
        """
        Convert a list[list[ProbsMetrics]] into a 3D-like DataFrame.

        Structure:
        - metrics[layer][head] -> ProbsMetrics with 18 metrics

        Returns DataFrame with MultiIndex:
        - Index level 0: Layer (0, 1, 2, ...)
        - Index level 1: Head (0, 1, 2, ...)
        - Columns: Metric names (KL, JS, TV, L2, etc.)
        """
        data = []
        for layer_idx, layer_metrics in enumerate(metrics):
            for head_idx, prob_metrics in enumerate(layer_metrics):
                metric_dict = prob_metrics.summary()
                metric_dict["layer"] = layer_idx
                metric_dict["head"] = head_idx
                data.append(metric_dict)

        df = pd.DataFrame(data)
        df = df.set_index(["layer", "head"])
        return cls(df)

    def get_metric(self, metric_name: str) -> pd.DataFrame:
        """
        Extract a specific metric across all layers and heads.
        Returns a 2D DataFrame: rows=layers, columns=heads
        """
        if metric_name not in self.df.columns:
            available = list(self.df.columns)
            raise ValueError(
                f"Metric '{metric_name}' not found. Available: {available}"
            )

        metric_data = self.df[metric_name].unstack(level="head")
        return metric_data

    def get_layer(self, layer_idx: int) -> pd.DataFrame:
        """Get all metrics for a specific layer across all heads."""
        layer = self.df.loc[layer_idx]
        assert isinstance(layer, pd.DataFrame)
        return layer

    def get_head(self, layer_idx: int, head_idx: int) -> pd.Series:
        """Get all metrics for a specific head"""
        layer = self.get_layer(layer_idx)
        head = layer.loc[head_idx]
        assert isinstance(head, pd.Series)
        return head

    def summary_stats(self) -> pd.DataFrame:
        """Get summary statistics across all layers and heads for each metric."""
        return self.df.describe()

    def plot_kl(
        self,
        fig: go.Figure,
        row: int | None = None,
        col: int | None = None,
        colorbar: dict | None = None,
    ) -> go.Figure:
        """
        Plot KL divergence heatmap.

        Args:
            fig: plotly figure to add to (required)
            row: Row position for subplot (1-indexed)
            col: Column position for subplot (1-indexed)
            colorbar: Optional colorbar configuration dictionary.

        Returns:
            plotly.graph_objects.Figure
        """
        kl_pd = self.get_metric("KL")

        # Add as subplot
        heatmap = go.Heatmap(
            z=kl_pd.values,
            x=kl_pd.columns,
            y=kl_pd.index,
            text=kl_pd.round(4).values,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="reds",
            name="KL Divergence",
            colorbar=colorbar,
        )
        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, row=row, col=col)
        fig.update_yaxes(
            tickmode="linear",
            tick0=0,
            dtick=1,
            row=row,
            col=col,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )

        return fig

    def plot_s1_factor(
        self,
        fig: go.Figure,
        row: int | None = None,
        col: int | None = None,
        colorbar: dict | None = None,
    ) -> go.Figure:
        """
        Plot S1 Prob Factor heatmap.

        Args:
            fig: plotly figure to add to (required)
            row: Row position for subplot (1-indexed)
            col: Column position for subplot (1-indexed)
            colorbar: Optional colorbar configuration dictionary.

        Returns:
            plotly.graph_objects.Figure
        """
        s1_prob_factor_df = self.get_metric("s1_prob_factor")

        # Add as subplot
        heatmap = go.Heatmap(
            z=s1_prob_factor_df.values,
            x=s1_prob_factor_df.columns,
            y=s1_prob_factor_df.index,
            text=s1_prob_factor_df.round(4).values,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="blues",
            name="S1 Prob Factor",
            colorbar=colorbar,
        )
        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, row=row, col=col)
        fig.update_yaxes(
            tickmode="linear",
            tick0=0,
            dtick=1,
            row=row,
            col=col,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )

        return fig


"""
Example Usage:

# Assume you have metrics: list[list[ProbsMetrics]]
# where metrics[layer][head] gives you a ProbsMetrics object

# Create the 3D-like DataFrame
plotter = Plotter.from_metrics(metrics)

# Access the MultiIndex DataFrame
df_3d = plotter.df
print(df_3d.shape)  # (num_layers * num_heads, 18 metrics)

# Get a specific metric across all layers and heads as a 2D matrix
kl_matrix = plotter.get_metric('KL')  # Shape: (num_layers, num_heads)
js_matrix = plotter.get_metric('JS')

# Get all metrics for layer 0 (across all heads)
layer_0_data = plotter.get_layer(0)  # Shape: (num_heads, 18 metrics)

# Get all metrics for head 1 (across all layers)  
head_1_data = plotter.get_head(1)  # Shape: (num_layers, 18 metrics)

# Summary statistics across all layers/heads
stats = plotter.summary_stats()

# Wide format (alternative representation)
plotter_wide = Plotter.from_metrics_wide(metrics)
df_wide = plotter_wide.df  # Shape: (18 metrics, num_layers * num_heads)

# Accessing specific combinations:
# Layer 2, Head 5, KL divergence
kl_value = plotter.df.loc[(2, 5), 'KL']

# All KL values as a flat series
all_kl = plotter.df['KL']

# Plotting examples:
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of KL divergence across layers and heads
plt.figure(figsize=(10, 6))
sns.heatmap(plotter.get_metric('KL'), annot=True, cmap='viridis')
plt.title('KL Divergence: Layers vs Heads')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.show()
"""
