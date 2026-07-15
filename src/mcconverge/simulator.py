import numpy as np
from typing import Callable
from mcconverge.variables import RandomVariable

class SimulationResult:
    """
    Container for the results of a Monte Carlo simulation.

    Parameters
    ----------
    outputs : np.ndarray
        Array of model output values (one per sample).
    inputs : dict[str, np.ndarray]
        Dictionary mapping input variable names to their sampled arrays.

    Examples
    --------
    >>> result.mean()
    2300000.0
    >>> result.percentile(5)
    400000.0
    """

    def __init__(self, outputs: np.ndarray, inputs: dict):
        self.outputs = outputs
        self.inputs = inputs

    def mean(self) -> float:
        """Return the mean of the output distribution."""
        return float(np.mean(self.outputs))

    def std(self) -> float:
        """Return the standard deviation of the output distribution."""
        return float(np.std(self.outputs))

    def percentile(self, p: float) -> float:
        """
        Return the p-th percentile of the output distribution.

        Parameters
        ----------
        p : float
            Percentile between 0 and 100.
        """
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        return float(np.percentile(self.outputs, p))

    def confidence_interval(self, level: float = 0.95) -> tuple:
        """
        Return a symmetric confidence interval around the mean.

        Parameters
        ----------
        level : float
            Confidence level between 0 and 1. Default is 0.95.

        Returns
        -------
        tuple
            (lower bound, upper bound)
        """
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        lower = (1 - level) / 2 * 100
        upper = (1 + level) / 2 * 100
        return (self.percentile(lower), self.percentile(upper))

    def summary(self) -> dict:
        """Return a dictionary of key statistics."""
        return {
            "mean":   self.mean(),
            "std":    self.std(),
            "p5":     self.percentile(5),
            "p95":    self.percentile(95),
            "min":    float(np.min(self.outputs)),
            "max":    float(np.max(self.outputs)),
            "n":      len(self.outputs),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"SimulationResult(n={s['n']}, mean={s['mean']:.2f}, "
                f"std={s['std']:.2f}, 90% CI=[{s['p5']:.2f}, {s['p95']:.2f}])")


