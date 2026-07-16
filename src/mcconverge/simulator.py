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


class MonteCarloSimulator:
    """
    General-purpose Monte Carlo simulator.

    Runs a user-supplied model f(X) N times with sampled inputs
    and collects the output distribution.

    Parameters
    ----------
    model : Callable
        A function that takes a dict of {name: sampled_value} and returns a float.
    n_samples : int
        Number of Monte Carlo samples. Default is 10000.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> def my_model(samples):
    ...     return samples["price"] * samples["quantity"]
    >>> sim = MonteCarloSimulator(model=my_model, n_samples=10000, seed=42)
    >>> sim.add_input(RandomVariable("price", "normal", mean=10, std=2))
    >>> sim.add_input(RandomVariable("quantity", "uniform", low=50, high=150))
    >>> result = sim.run()
    """

    def __init__(self, model: Callable, n_samples: int = 10_000, seed: int = None):
        if not callable(model):
            raise ValueError("model must be a callable")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self.model = model
        self.n_samples = n_samples
        self.seed = seed
        self.inputs: dict[str, RandomVariable] = {}

    def add_input(self, rv: RandomVariable) -> None:
        """
        Register a RandomVariable as an input to the model.

        Parameters
        ----------
        rv : RandomVariable
            The uncertain input variable to add.
        """
        if not isinstance(rv, RandomVariable):
            raise TypeError("rv must be a RandomVariable instance")
        self.inputs[rv.name] = rv

    def run(self) -> SimulationResult:
        """
        Run the Monte Carlo simulation.

        Samples all input variables, evaluates the model for each
        sample, and returns a SimulationResult.

        Returns
        -------
        SimulationResult
        """
        if not self.inputs:
            raise RuntimeError("No input variables registered. Use add_input() first.")

        rng = np.random.default_rng(self.seed)

        # Sample all inputs — builds the N x K matrix from the lecture
        sampled_inputs = {
            name: rv.sample(self.n_samples, rng)
            for name, rv in self.inputs.items()
        }

        # Evaluate the model for each of the N scenarios
        outputs = np.array([
            self.model({name: sampled_inputs[name][i] for name in sampled_inputs})
            for i in range(self.n_samples)
        ])

        return SimulationResult(outputs=outputs, inputs=sampled_inputs)

    def run_convergence_study(
        self,
        sample_sizes: list = None,
        metric: str = "mean",
        n_repetitions: int = 10,
    ) -> "ConvergenceResult":
        """
        Run the simulation at increasing sample sizes to study convergence.

        Parameters
        ----------
        sample_sizes : list of int, optional
            Sample sizes to test. Defaults to log-spaced values from 10 to n_samples.
        metric : str
            Statistic to track. One of: "mean", "std", "p5", "p95".
        n_repetitions : int
            Number of times to repeat each sample size to estimate spread.

        Returns
        -------
        ConvergenceResult
        """
        supported_metrics = ["mean", "std", "p5", "p95"]
        if metric not in supported_metrics:
            raise ValueError(f"metric must be one of {supported_metrics}")

        if sample_sizes is None:
            sample_sizes = [int(x) for x in np.logspace(1, np.log10(self.n_samples), 20)]
            sample_sizes = sorted(set(sample_sizes))

        estimates = []   # shape: (len(sample_sizes), n_repetitions)

        for n in sample_sizes:
            reps = []
            for _ in range(n_repetitions):
                rng = np.random.default_rng()   # fresh seed each rep
                sampled = {
                    name: rv.sample(n, rng)
                    for name, rv in self.inputs.items()
                }
                outputs = np.array([
                    self.model({name: sampled[name][i] for name in sampled})
                    for i in range(n)
                ])
                result = SimulationResult(outputs=outputs, inputs=sampled)
                metric_map = {
                    "mean": result.mean(),
                    "std":  result.std(),
                    "p5":   result.percentile(5),
                    "p95":  result.percentile(95),
                }
                reps.append(metric_map[metric])
            estimates.append(reps)

        return ConvergenceResult(
            sample_sizes=sample_sizes,
            estimates=np.array(estimates),
            metric=metric,
        )


class ConvergenceResult:
    """
    Holds the results of a convergence study.

    Parameters
    ----------
    sample_sizes : list of int
        The sample sizes tested.
    estimates : np.ndarray
        Shape (len(sample_sizes), n_repetitions). Each row contains
        repeated estimates of the metric at that sample size.
    metric : str
        The metric that was tracked (e.g. "mean").
    """

    def __init__(self, sample_sizes: list, estimates: np.ndarray, metric: str):
        self.sample_sizes = sample_sizes
        self.estimates = estimates
        self.metric = metric

    def recommended_n(self, tolerance: float = 0.01) -> int:
        """
        Suggest the minimum sample size where the estimate has stabilized.

        Stability is defined as the point where the spread (std across
        repetitions) falls below tolerance * mean estimate.

        Parameters
        ----------
        tolerance : float
            Relative tolerance. Default 0.01 means 1% of the mean estimate.

        Returns
        -------
        int
            Recommended minimum number of samples.
        """
        overall_mean = np.mean(self.estimates)
        for i, n in enumerate(self.sample_sizes):
            spread = np.std(self.estimates[i])
            if spread < tolerance * abs(overall_mean):
                return n
        return self.sample_sizes[-1]

    def __repr__(self) -> str:
        return (f"ConvergenceResult(metric='{self.metric}', "
                f"sample_sizes={self.sample_sizes[0]}..{self.sample_sizes[-1]}, "
                f"recommended_n={self.recommended_n()})")


class CostComponent:
    """
    Represents a single cost line item in an NPV model.

    The annual cost is computed as:
        cost_t = quantity * unit_price * (1 + escalation_rate)^t

    Parameters
    ----------
    name : str
        Name of the cost item (e.g. "electricity").
    quantity : float or RandomVariable
        Amount consumed per year (e.g. kWh/year).
    unit_price : float or RandomVariable
        Price per unit (e.g. €/kWh).
    escalation_rate : float
        Annual price increase rate. Default 0.0.

    Examples
    --------
    >>> elec_price = RandomVariable("electricity", "normal", mean=0.12, std=0.02)
    >>> elec = CostComponent("Electricity", quantity=800, unit_price=elec_price)
    >>> elec.annual_cost(year=5, samples={"electricity": 0.13})
    104.0
    """

    def __init__(
        self,
        name: str,
        quantity,
        unit_price,
        escalation_rate: float = 0.0,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if escalation_rate < 0:
            raise ValueError("escalation_rate must be non-negative")
        self.name = name
        self.quantity = quantity
        self.unit_price = unit_price
        self.escalation_rate = escalation_rate

    def annual_cost(self, year: int, samples: dict) -> float:
        """
        Compute the cost for a given year and set of sampled values.

        Parameters
        ----------
        year : int
            The year in the investment horizon (0-indexed).
        samples : dict
            Dictionary of sampled input values from the simulator.

        Returns
        -------
        float
            Cost for this component in the given year.
        """
        q = samples[self.quantity.name] if isinstance(self.quantity, RandomVariable) else self.quantity
        p = samples[self.unit_price.name] if isinstance(self.unit_price, RandomVariable) else self.unit_price
        return q * p * (1 + self.escalation_rate) ** year

    def __repr__(self) -> str:
        return f"CostComponent(name='{self.name}', escalation_rate={self.escalation_rate})"


class NPVModel:
    """
    Net Present Value (Barwert) model.

    Aggregates multiple CostComponents and discounts their cash flows
    to compute the NPV over an investment horizon.

    Parameters
    ----------
    horizon_years : int
        Number of years in the investment horizon.
    discount_rate : float or RandomVariable
        Annual discount rate (e.g. 0.05 for 5%).

    Examples
    --------
    >>> model = NPVModel(horizon_years=20, discount_rate=0.05)
    >>> model.add_cost(CostComponent("CO2", quantity=5.0, unit_price=co2_rv))
    >>> npv = model(samples)
    """

    def __init__(self, horizon_years: int, discount_rate):
        if not isinstance(horizon_years, int) or horizon_years <= 0:
            raise ValueError("horizon_years must be a positive integer")
        self.horizon_years = horizon_years
        self.discount_rate = discount_rate
        self._costs: list[CostComponent] = []
        self._revenues: list[CostComponent] = []

    def add_cost(self, component: CostComponent) -> None:
        """Add a cost component to the model."""
        if not isinstance(component, CostComponent):
            raise TypeError("component must be a CostComponent instance")
        self._costs.append(component)

    def add_revenue(self, component: CostComponent) -> None:
        """Add a revenue component to the model."""
        if not isinstance(component, CostComponent):
            raise TypeError("component must be a CostComponent instance")
        self._revenues.append(component)

    def __call__(self, samples: dict) -> float:
        """
        Evaluate the NPV for one set of sampled input values.

        Parameters
        ----------
        samples : dict
            Dictionary of {variable_name: sampled_value}.

        Returns
        -------
        float
            Net Present Value for this scenario.
        """
        r = (samples[self.discount_rate.name]
             if isinstance(self.discount_rate, RandomVariable)
             else self.discount_rate)

        npv = 0.0
        for t in range(self.horizon_years):
            total_cost = sum(c.annual_cost(t, samples) for c in self._costs)
            total_revenue = sum(rev.annual_cost(t, samples) for rev in self._revenues)
            cash_flow = total_revenue - total_cost
            npv += cash_flow / (1 + r) ** t

        return npv

    def __repr__(self) -> str:
        return (f"NPVModel(horizon_years={self.horizon_years}, "
                f"costs={len(self._costs)}, revenues={len(self._revenues)})")