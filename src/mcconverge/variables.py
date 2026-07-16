import numpy as np

SUPPORTED_DISTRIBUTIONS = ["normal", "uniform", "triangular", "lognormal"]

class RandomVariable:
    """
    Represents an uncertain input parameter with a probability distribution.

    Parameters
    ----------
    name : str
        Name of the variable (e.g. "co2_price").
    distribution : str
        Type of distribution. Must be one of: normal, uniform, triangular, lognormal.
    **params : dict
        Distribution parameters:
        - normal:      mean, std
        - uniform:     low, high
        - triangular:  low, mode, high
        - lognormal:   mean, std (of the underlying normal)

    Examples
    --------
    >>> rv = RandomVariable("co2_price", "triangular", low=20, mode=80, high=200)
    >>> samples = rv.sample(1000)
    """

    def __init__(self, name: str, distribution: str, **params):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution '{distribution}' is not supported. "
                f"Choose from: {SUPPORTED_DISTRIBUTIONS}"
            )
        self.name = name
        self.distribution = distribution
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        """Check that the required parameters for the chosen distribution are provided."""
        required = {
            "normal":     ["mean", "std"],
            "uniform":    ["low", "high"],
            "triangular": ["low", "mode", "high"],
            "lognormal":  ["mean", "std"],
        }
        for key in required[self.distribution]:
            if key not in self.params:
                raise ValueError(
                    f"Distribution '{self.distribution}' requires parameter '{key}'"
                )
        if self.distribution in ["normal", "lognormal"]:
            if self.params["std"] <= 0:
                raise ValueError("std must be positive")
        if self.distribution in ["uniform", "triangular"]:
            if self.params["low"] >= self.params["high"]:
                raise ValueError("low must be less than high")
        if self.distribution == "triangular":
            if not (self.params["low"] <= self.params["mode"] <= self.params["high"]):
                raise ValueError("mode must be between low and high")

    def sample(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """
        Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of n sampled values.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if rng is None:
            rng = np.random.default_rng()

        p = self.params
        if self.distribution == "normal":
            return rng.normal(p["mean"], p["std"], n)
        elif self.distribution == "uniform":
            return rng.uniform(p["low"], p["high"], n)
        elif self.distribution == "triangular":
            return rng.triangular(p["low"], p["mode"], p["high"], n)
        elif self.distribution == "lognormal":
            return rng.lognormal(p["mean"], p["std"], n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density function at given points.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            PDF values at each point in x.
        """
        from scipy import stats
        p = self.params
        if self.distribution == "normal":
            return stats.norm.pdf(x, p["mean"], p["std"])
        elif self.distribution == "uniform":
            return stats.uniform.pdf(x, p["low"], p["high"] - p["low"])
        elif self.distribution == "triangular":
            span = p["high"] - p["low"]
            c = (p["mode"] - p["low"]) / span
            return stats.triang.pdf(x, c, loc=p["low"], scale=span)
        elif self.distribution == "lognormal":
            return stats.lognorm.pdf(x, s=p["std"], scale=np.exp(p["mean"]))

    def __repr__(self) -> str:
        return f"RandomVariable(name='{self.name}', distribution='{self.distribution}', params={self.params})"