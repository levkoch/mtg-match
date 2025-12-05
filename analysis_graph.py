import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_hyperparameter_loss(
        hyperparameter_values: list[float], losses: list[float], chart_title: str, 
        xlabel: str = "Hyperparameter Value", ylabel: str= "Good Detection Rate", 
        figsize: tuple[int, int] = (10, 6), color: str = 'steelblue'):
    """
    Plot average loss for each hyperparameter value with optional polynomial fit.
    
    Args:
        hyperparameter_values: List of hyperparameter values (can have duplicates)
        losses: List of loss values corresponding to each hyperparameter
        chart_title: Title for the chart
        xlabel: Label for x-axis (defaults to "Hyperparameter Value")
        ylabel: Label for y-axis (defaults to "Good Detection Rate")
        figsize: Tuple of (width, height) for figure size
        color: Color for the bars/points
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    
    if len(hyperparameter_values) != len(losses):
        raise ValueError("hyperparameter_values and losses must have the same length")
    
    # aggregate losses and find mean and std
    loss_dict = defaultdict(list)
    for param, loss in zip(hyperparameter_values, losses):
        loss_dict[param].append(loss)
    
    params_sorted = sorted(loss_dict.keys())
    avg_losses = [np.mean(loss_dict[param]) for param in params_sorted]
    std_losses = [np.std(loss_dict[param]) for param in params_sorted]
    
    fig, ax = plt.subplots(figsize=figsize)

    # line plot with error bars
    ax.errorbar(params_sorted, avg_losses, yerr=std_losses, 
                marker='o', linewidth=2, markersize=8, 
                color=color, capsize=5, capthick=2, 
                ecolor=color, alpha=1.0, elinewidth=1.5,
                zorder=3)
    
    # make spread bars lighter
    for line in ax.lines[1:]:
        line.set_alpha(0.5)
    for cap in ax.collections:
        cap.set_alpha(0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(chart_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    plt.tight_layout()
    return fig, ax
