import math
from typing import Self

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    def plot_metric(
        self,
        fig: go.Figure,
        metric_name: str,
        name: str | None = None,
        colorscale: str = "blues",
        row: int | None = None,
        col: int | None = None,
        colorbar: dict | None = None,
        show_colorbar: bool = True,
    ) -> go.Figure:
        """
        Generic function to plot any metric as a heatmap.

        Args:
            fig: plotly figure to add to (required)
            metric_name: The name of the metric to retrieve from the DataFrame.
            name: Optional display name for the trace. If None, generated from metric_name.
            colorscale: Optional colorscale name. If None, a default is chosen.
            row: Row position for subplot (1-indexed).
            col: Column position for subplot (1-indexed).
            colorbar: Optional colorbar configuration dictionary.
            show_colorbar: Whether to display the colorbar for this heatmap.

        Returns:
            plotly.graph_objects.Figure
        """
        metric_df = self.get_metric(metric_name)

        # Auto-generate a display name if not provided
        display_name = name or metric_name.replace("_", " ").title()

        heatmap = go.Heatmap(
            z=metric_df.values,
            x=metric_df.columns,
            y=metric_df.index,
            text=metric_df.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            name=display_name,
            colorbar=colorbar,
            showscale=show_colorbar,
        )

        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, row=row, col=col)
        fig.update_yaxes(
            tickmode="linear",
            tick0=0,
            dtick=1,
            row=row,
            col=col,
        )

        return fig

    def plot_metrics(
        self,
        metric_names: list[str],
        cols: int | None = None,
        show_colorbars: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Plots a grid of heatmaps for a given list of metric names.

        Args:
            metric_names: A list of metric names to plot.
            cols: Optional number of columns in the subplot grid. If None,
                  a reasonable default is used (3 or less).
            show_colorbars: Whether to display the colorbars for the heatmaps.
            **kwargs: Additional keyword arguments passed to make_subplots
                      (e.g., horizontal_spacing, vertical_spacing).

        Returns:
            A plotly.graph_objects.Figure with the generated subplots.
        """
        if not metric_names:
            return go.Figure()

        n_metrics = len(metric_names)
        n_cols = min(cols, n_metrics) if cols is not None else min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)

        subplot_titles = [name.replace("_", " ").title() for name in metric_names]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            **kwargs,
        )

        for i, metric_name in enumerate(metric_names):
            row = i // n_cols + 1
            col = i % n_cols + 1
            self.plot_metric(
                fig,
                metric_name=metric_name,
                row=row,
                col=col,
                show_colorbar=show_colorbars,
            )

        fig.update_layout(height=n_rows * 400, title_text="Metric Analysis")
        return fig
