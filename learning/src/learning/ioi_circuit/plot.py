import math
from typing import Self

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from learning.ioi_circuit.metrics import ProbsMetrics


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
        metric_name: str,
        fig: go.Figure,
        row: int,
        col: int,
        colorscale: str,
        colorbar: dict,
        show_colorbar: bool = True,
        center_at_zero: bool = True,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> go.Figure:
        """
        Plot a single metric as a heatmap.

        Args:
            metric_name: metric column name to retrieve from the DataFrame.
            fig: plotly figure to draw on.
            row: row position for subplot (1-indexed).
            col: column position for subplot (1-indexed).
            colorscale: colorscale name (e.g., "RdBu").
            colorbar: Optional colorbar configuration dictionary.
            show_colorbar: Whether to display the colorbar for this heatmap.
            center_at_zero: If True, the colorscale will be centered at 0.
            zmin: The minimum value of the colorscale.
            zmax: The maximum value of the colorscale.
        """
        metric_df = self.get_metric(metric_name)

        heatmap_args: dict = dict(
            z=metric_df.values,
            x=metric_df.columns,
            y=metric_df.index,
            text=metric_df.round(4).values,
            texttemplate="%{text}",
            colorscale=colorscale,
            name=metric_name,
            colorbar=colorbar,
            showscale=show_colorbar,
        )
        if center_at_zero:
            heatmap_args["zmid"] = 0

        if zmin is not None:
            heatmap_args["zmin"] = zmin
        if zmax is not None:
            heatmap_args["zmax"] = zmax

        heatmap = go.Heatmap(**heatmap_args)

        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(
            tickmode="linear",
            tick0=0,
            dtick=1,
            row=row,
            col=col,
            ticklabelstandoff=5,
        )
        fig.update_yaxes(
            tickmode="linear",
            tick0=0,
            dtick=1,
            row=row,
            col=col,
            ticklabelstandoff=5,
        )

        return fig

    def plot_metrics(
        self,
        metric_names: list[str],
        cols: int = 1,
        row_height: float = 400,
        colorscale: str = "blues",
        uniform_colorscale: bool = False,
        center_at_zero: bool = False,
    ) -> go.Figure:
        """
        Plots a grid of heatmaps for a given list of metric names.

        Rows are layers, columns are heads.

        Args:
            metric_names: A list of metric names to plot.
            cols: number of columns in the subplot grid.
            colorscale: The colorscale to use for the heatmaps.
            uniform_colorscale: If True, all subplots will share the same colorscale.
            center_at_zero: If True, the colorscale will be centered at 0.
        """
        if not metric_names:
            return go.Figure()

        # If uniform_colorscale, compute the min and max values for the colorscale.
        zmin, zmax = None, None
        if uniform_colorscale:
            all_metrics_df = self.df[metric_names]
            min_val = all_metrics_df.min().min()
            max_val = all_metrics_df.max().max()

            if center_at_zero:
                abs_max = max(abs(min_val), abs(max_val))
                zmin, zmax = -abs_max, abs_max
            else:
                zmin, zmax = min_val, max_val

        # Compute the number of rows and columns in the subplot grid.
        n_metrics = len(metric_names)
        n_cols = min(cols, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)

        # Compute the horizontal and vertical spacing between subplots.
        horizontal_spacing = 0.2 / n_cols if n_cols > 1 else 0
        vertical_spacing = 0.2 / n_rows if n_rows > 1 else 0

        # Calculate subplot width and height based on spacing
        subplot_width = (1.0 - (n_cols - 1) * horizontal_spacing) / n_cols
        subplot_height = (1.0 - (n_rows - 1) * vertical_spacing) / n_rows

        # Create the subplot grid.
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=metric_names,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )

        for i, metric_name in enumerate(metric_names):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Calculate the x position for the colorbar
            colorbar_x = (col - 1) * (
                subplot_width + horizontal_spacing
            ) + subplot_width

            # Calculate colorbar y position and height
            y_top = 1.0 - (row - 1) * (subplot_height + vertical_spacing)
            y_bottom = y_top - subplot_height
            colorbar_y = y_bottom + subplot_height / 2
            colorbar_len = subplot_height + vertical_spacing / 2

            colorbar_opts = dict(
                x=colorbar_x,
                xanchor="left",
                thickness=10,
                y=colorbar_y,
                yanchor="middle",
                len=colorbar_len,
            )

            self.plot_metric(
                metric_name=metric_name,
                fig=fig,
                row=row,
                col=col,
                colorscale=colorscale,
                colorbar=colorbar_opts,
                center_at_zero=center_at_zero,
                zmin=zmin,
                zmax=zmax,
            )

        fig.update_layout(height=n_rows * row_height)
        return fig
