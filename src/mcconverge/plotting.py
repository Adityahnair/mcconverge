import numpy as np
import matplotlib.pyplot as plt
from mcconverge.simulator import SimulationResult, ConvergenceResult
from scipy.stats import gaussian_kde

def plot_output_distribution(
    result: SimulationResult,
    bins: int = 50,
    show_percentiles: bool = True,
    title: str = "Output Distribution",
    xlabel: str = "Value",
) -> plt.Figure:
    """
    Plot the histogram of simulation outputs with optional percentile markers.

    Parameters
    ----------
    result : SimulationResult
        Result object from MonteCarloSimulator.run().
    bins : int
        Number of histogram bins. Default 50.
    show_percentiles : bool
        If True, mark the 5th and 95th percentiles. Default True.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(result.outputs, bins=bins, density=True,
            color="steelblue", alpha=0.7, edgecolor="white", label="Simulated outputs")

    if show_percentiles:
        p5  = result.percentile(5)
        p95 = result.percentile(95)
        ax.axvline(p5,  color="red",    linestyle="--", linewidth=1.5, label=f"5th percentile: {p5:.2f}")
        ax.axvline(p95, color="orange", linestyle="--", linewidth=1.5, label=f"95th percentile: {p95:.2f}")
        ax.axvline(result.mean(), color="black", linestyle="-", linewidth=1.5, label=f"Mean: {result.mean():.2f}")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_input_distribution(rv, n_samples: int = 10_000, seed: int = None) -> plt.Figure:
    """
    Plot the distribution of a RandomVariable, showing both
    sampled histogram and theoretical PDF.

    Parameters
    ----------
    rv : RandomVariable
        The input variable to plot.
    n_samples : int
        Number of samples to draw for the histogram.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(seed)
    samples = rv.sample(n_samples, rng)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=50, density=True,
            color="steelblue", alpha=0.6, edgecolor="white", label="Samples")

    x = np.linspace(samples.min(), samples.max(), 300)
    ax.plot(x, rv.pdf(x), color="darkblue", linewidth=2, label="Theoretical PDF")

    ax.set_title(f"Distribution of '{rv.name}'", fontsize=14)
    ax.set_xlabel(rv.name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_convergence(conv_result: ConvergenceResult) -> plt.Figure:
    """
    Plot how a statistic converges as sample size increases.

    Shows the mean estimate across repetitions and a shaded band
    representing ± 1 standard deviation across repetitions.

    Parameters
    ----------
    conv_result : ConvergenceResult
        Result from MonteCarloSimulator.run_convergence_study().

    Returns
    -------
    matplotlib.figure.Figure
    """
    sizes  = conv_result.sample_sizes
    means  = np.mean(conv_result.estimates, axis=1)
    stds   = np.std(conv_result.estimates, axis=1)
    rec_n  = conv_result.recommended_n()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sizes, means, color="steelblue", linewidth=2, marker="o",
            markersize=4, label=f"Mean estimate ({conv_result.metric})")
    ax.fill_between(sizes, means - stds, means + stds,
                    color="steelblue", alpha=0.2, label="± 1 std across repetitions")
    ax.axvline(rec_n, color="red", linestyle="--", linewidth=1.5,
               label=f"Recommended N = {rec_n}")

    ax.set_xscale("log")
    ax.set_title(f"Convergence of '{conv_result.metric}'", fontsize=14)
    ax.set_xlabel("Number of samples (log scale)", fontsize=12)
    ax.set_ylabel(conv_result.metric.capitalize(), fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_tornado(result: SimulationResult, title: str = "Sensitivity Analysis") -> plt.Figure:
    """
    Plot a tornado chart showing Spearman rank correlations between
    each input variable and the output.

    Parameters
    ----------
    result : SimulationResult
        Result from MonteCarloSimulator.run().
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import spearmanr

    correlations = {}
    for name, input_samples in result.inputs.items():
        corr, _ = spearmanr(input_samples, result.outputs)
        correlations[name] = corr

    # Sort by absolute correlation (largest at top — tornado shape)
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]))
    names  = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Spearman Rank Correlation with Output", fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=10)

    fig.tight_layout()
    return fig


def plot_input_vs_output(result: SimulationResult, input_name: str) -> plt.Figure:
    """
    Scatter plot of one input variable against the simulation output.

    Useful for visually identifying which inputs drive the output
    before running a formal sensitivity analysis.

    Parameters
    ----------
    result : SimulationResult
        Result from MonteCarloSimulator.run().
    input_name : str
        Name of the input variable to plot on the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if input_name not in result.inputs:
        raise ValueError(f"'{input_name}' not found in simulation inputs. "
                         f"Available: {list(result.inputs.keys())}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result.inputs[input_name], result.outputs,
               alpha=0.2, s=5, color="steelblue")
    ax.set_xlabel(input_name, fontsize=12)
    ax.set_ylabel("Model Output", fontsize=12)
    ax.set_title(f"'{input_name}' vs Output", fontsize=14)
    fig.tight_layout()
    return fig

def plot_simulation_input(result: SimulationResult, input_name: str) -> plt.Figure:
    """
    Plot the actual input samples used in the Monte Carlo simulation
    as a histogram with KDE overlay.

    Parameters
    ----------
    result : SimulationResult
        Result object from MonteCarloSimulator.run().
    input_name : str
        Name of the input variable to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if input_name not in result.inputs:
        raise ValueError(
            f"'{input_name}' not found in simulation inputs. "
            f"Available: {list(result.inputs.keys())}"
        )

    samples = result.inputs[input_name]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=50, density=True,
            color="steelblue", alpha=0.6, edgecolor="white", label="Simulation samples")

    kde = gaussian_kde(samples)
    x = np.linspace(samples.min(), samples.max(), 300)
    ax.plot(x, kde(x), color="darkblue", linewidth=2, label="KDE")

    ax.axvline(samples.mean(), color="black", linestyle="--",
               linewidth=1.5, label=f"Mean: {samples.mean():.4f}")

    ax.set_title(f"Actual simulation input: '{input_name}'", fontsize=14)
    ax.set_xlabel(input_name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cost_breakdown(result: SimulationResult, model, title: str = "Cost Breakdown") -> plt.Figure:
    """
    Plot the average discounted cost contribution of each cost component
    as a horizontal bar chart.

    Parameters
    ----------
    result : SimulationResult
        Result object from MonteCarloSimulator.run().
    model : NPVModel
        The NPV model containing the cost components.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    contributions = {}

    for component in model._costs:
        # Compute average discounted cost over all years and all samples
        total = 0.0
        for t in range(model.horizon_years):
            # Use mean of each input sample for the average cost
            mean_samples = {name: float(np.mean(arr)) for name, arr in result.inputs.items()}
            total += component.annual_cost(t, mean_samples) / (1 + model.discount_rate) ** t
        contributions[component.name] = total

    total_cost = sum(contributions.values())
    percentages = {k: (v / total_cost) * 100 for k, v in contributions.items()}

    # Sort by contribution size
    sorted_items = sorted(percentages.items(), key=lambda x: x[1])
    names  = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
    bars = ax.barh(names, values, color="steelblue", alpha=0.7, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Share of total discounted cost (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig