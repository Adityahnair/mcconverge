from mcconverge import mcconverge

import numpy as np
import pytest
from mcconverge.variables import RandomVariable, SUPPORTED_DISTRIBUTIONS
from mcconverge.simulator import (
    SimulationResult, MonteCarloSimulator,
    CostComponent, NPVModel, ConvergenceResult
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcconverge import plotting

# ─── RandomVariable ───────────────────────────────────────────────────────────

class TestRandomVariable:

    def test_valid_normal(self):
        rv = RandomVariable("x", "normal", mean=0, std=1)
        assert rv.name == "x"
        assert rv.distribution == "normal"

    def test_valid_uniform(self):
        rv = RandomVariable("x", "uniform", low=0, high=10)
        assert rv.distribution == "uniform"

    def test_valid_triangular(self):
        rv = RandomVariable("x", "triangular", low=0, mode=5, high=10)
        assert rv.distribution == "triangular"

    def test_valid_lognormal(self):
        rv = RandomVariable("x", "lognormal", mean=1, std=0.5)
        assert rv.distribution == "lognormal"

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="not supported"):
            RandomVariable("x", "exponential", mean=1)

    def test_empty_name(self):
        with pytest.raises(ValueError):
            RandomVariable("", "normal", mean=0, std=1)

    def test_missing_param(self):
        with pytest.raises(ValueError, match="requires parameter"):
            RandomVariable("x", "normal", mean=0)   # missing std

    def test_negative_std(self):
        with pytest.raises(ValueError, match="std must be positive"):
            RandomVariable("x", "normal", mean=0, std=-1)

    def test_low_greater_than_high(self):
        with pytest.raises(ValueError, match="low must be less than high"):
            RandomVariable("x", "uniform", low=10, high=5)

    def test_mode_outside_range(self):
        with pytest.raises(ValueError, match="mode must be between"):
            RandomVariable("x", "triangular", low=0, mode=15, high=10)

    def test_sample_returns_correct_shape(self):
        rv = RandomVariable("x", "normal", mean=0, std=1)
        samples = rv.sample(500)
        assert samples.shape == (500,)

    def test_sample_invalid_n(self):
        rv = RandomVariable("x", "normal", mean=0, std=1)
        with pytest.raises(ValueError):
            rv.sample(-10)

    def test_sample_mean_converges(self):
        """With large N, the sample mean should be close to the true mean."""
        rv = RandomVariable("x", "normal", mean=50, std=10)
        rng = np.random.default_rng(42)
        samples = rv.sample(100_000, rng)
        assert abs(samples.mean() - 50) < 0.1

    def test_sample_uniform_bounds(self):
        rv = RandomVariable("x", "uniform", low=5, high=15)
        rng = np.random.default_rng(0)
        samples = rv.sample(10_000, rng)
        assert samples.min() >= 5
        assert samples.max() <= 15

    def test_pdf_returns_correct_shape(self):
        rv = RandomVariable("x", "normal", mean=0, std=1)
        x = np.linspace(-3, 3, 100)
        pdf = rv.pdf(x)
        assert pdf.shape == (100,)

    def test_repr(self):
        rv = RandomVariable("price", "normal", mean=10, std=2)
        assert "price" in repr(rv)
        assert "normal" in repr(rv)


# ─── SimulationResult ─────────────────────────────────────────────────────────

class TestSimulationResult:

    def setup_method(self):
        """Create a simple result object for reuse across tests."""
        rng = np.random.default_rng(0)
        outputs = rng.normal(100, 10, 10_000)
        inputs  = {"x": rng.normal(0, 1, 10_000)}
        self.result = SimulationResult(outputs=outputs, inputs=inputs)

    def test_mean(self):
        assert abs(self.result.mean() - 100) < 0.5

    def test_std(self):
        assert abs(self.result.std() - 10) < 0.5

    def test_percentile_50_close_to_mean(self):
        assert abs(self.result.percentile(50) - self.result.mean()) < 1.0

    def test_percentile_invalid(self):
        with pytest.raises(ValueError):
            self.result.percentile(110)

    def test_confidence_interval_order(self):
        lo, hi = self.result.confidence_interval(0.95)
        assert lo < hi

    def test_confidence_interval_invalid_level(self):
        with pytest.raises(ValueError):
            self.result.confidence_interval(1.5)

    def test_summary_keys(self):
        s = self.result.summary()
        for key in ["mean", "std", "p5", "p95", "min", "max", "n"]:
            assert key in s

    def test_repr(self):
        assert "SimulationResult" in repr(self.result)


# ─── MonteCarloSimulator ──────────────────────────────────────────────────────

class TestMonteCarloSimulator:

    def setup_method(self):
        """Simple model: output = a * b"""
        self.model = lambda s: s["a"] * s["b"]
        self.rv_a  = RandomVariable("a", "uniform", low=1, high=3)
        self.rv_b  = RandomVariable("b", "normal",  mean=10, std=1)

    def test_run_returns_simulation_result(self):
        sim = MonteCarloSimulator(self.model, n_samples=1000, seed=0)
        sim.add_input(self.rv_a)
        sim.add_input(self.rv_b)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_output_length(self):
        sim = MonteCarloSimulator(self.model, n_samples=500, seed=0)
        sim.add_input(self.rv_a)
        sim.add_input(self.rv_b)
        result = sim.run()
        assert len(result.outputs) == 500

    def test_seed_reproducibility(self):
        sim1 = MonteCarloSimulator(self.model, n_samples=100, seed=42)
        sim1.add_input(self.rv_a); sim1.add_input(self.rv_b)

        sim2 = MonteCarloSimulator(self.model, n_samples=100, seed=42)
        sim2.add_input(self.rv_a); sim2.add_input(self.rv_b)

        np.testing.assert_array_equal(sim1.run().outputs, sim2.run().outputs)

    def test_no_inputs_raises(self):
        sim = MonteCarloSimulator(self.model, n_samples=100)
        with pytest.raises(RuntimeError, match="No input variables"):
            sim.run()

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="callable"):
            MonteCarloSimulator("not_a_function", n_samples=100)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError):
            MonteCarloSimulator(self.model, n_samples=-5)

    def test_add_invalid_input(self):
        sim = MonteCarloSimulator(self.model, n_samples=100)
        with pytest.raises(TypeError):
            sim.add_input("not_a_rv")

    def test_convergence_study_returns_convergence_result(self):
        sim = MonteCarloSimulator(self.model, n_samples=1000, seed=0)
        sim.add_input(self.rv_a)
        sim.add_input(self.rv_b)
        conv = sim.run_convergence_study(sample_sizes=[10, 50, 100], n_repetitions=3)
        assert isinstance(conv, ConvergenceResult)

    def test_convergence_invalid_metric(self):
        sim = MonteCarloSimulator(self.model, n_samples=500, seed=0)
        sim.add_input(self.rv_a)
        with pytest.raises(ValueError, match="metric"):
            sim.run_convergence_study(metric="median")


# ─── CostComponent ────────────────────────────────────────────────────────────

class TestCostComponent:

    def test_fixed_annual_cost(self):
        """With fixed quantity and price, cost = q * p * (1+r)^t"""
        comp = CostComponent("test", quantity=100, unit_price=2.0, escalation_rate=0.0)
        assert comp.annual_cost(year=0, samples={}) == pytest.approx(200.0)

    def test_escalation(self):
        comp = CostComponent("test", quantity=100, unit_price=2.0, escalation_rate=0.1)
        assert comp.annual_cost(year=1, samples={}) == pytest.approx(220.0)

    def test_rv_unit_price(self):
        rv = RandomVariable("elec", "normal", mean=0.12, std=0.01)
        comp = CostComponent("electricity", quantity=1000, unit_price=rv)
        cost = comp.annual_cost(year=0, samples={"elec": 0.15})
        assert cost == pytest.approx(150.0)

    def test_negative_escalation_rate(self):
        with pytest.raises(ValueError):
            CostComponent("test", quantity=1, unit_price=1, escalation_rate=-0.1)

    def test_empty_name(self):
        with pytest.raises(ValueError):
            CostComponent("", quantity=1, unit_price=1)


# ─── NPVModel ─────────────────────────────────────────────────────────────────

class TestNPVModel:

    def test_zero_cost_zero_npv(self):
        """With no components, NPV should be zero."""
        model = NPVModel(horizon_years=10, discount_rate=0.05)
        assert model({}) == pytest.approx(0.0)

    def test_deterministic_npv(self):
        """
        With fixed inputs and no discounting (r=0),
        NPV = annual_cash_flow * horizon_years.
        """
        model = NPVModel(horizon_years=5, discount_rate=0.0)
        model.add_revenue(CostComponent("rev", quantity=1000, unit_price=1.0))
        model.add_cost(CostComponent("cost", quantity=200, unit_price=1.0))
        # Expected: (1000 - 200) * 5 = 4000
        assert model({}) == pytest.approx(4000.0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError):
            NPVModel(horizon_years=-5, discount_rate=0.05)

    def test_add_invalid_cost(self):
        model = NPVModel(horizon_years=10, discount_rate=0.05)
        with pytest.raises(TypeError):
            model.add_cost("not_a_component")

    def test_callable_as_simulator_model(self):
        """NPVModel should work directly as the model argument in MonteCarloSimulator."""
        npv = NPVModel(horizon_years=5, discount_rate=0.05)
        npv.add_cost(CostComponent("energy", quantity=100, unit_price=2.0))
        sim = MonteCarloSimulator(model=npv, n_samples=100, seed=0)
        result = sim.run()
        assert isinstance(result, SimulationResult)


# ─── Shared fixture for plotting tests ────────────────────────────────────────

def make_result():
    """Helper to create a SimulationResult for plotting tests."""
    rng = np.random.default_rng(0)
    inputs = {
        "co2_price":   rng.triangular(20, 80, 200, 1000),
        "electricity": rng.normal(0.12, 0.03, 1000),
        "raw_material": rng.lognormal(4.5, 0.3, 1000),
    }
    outputs = (
        inputs["co2_price"] * 5.0
        + inputs["electricity"] * 800
        + inputs["raw_material"] * 1.1
    )
    return SimulationResult(outputs=outputs, inputs=inputs)


def make_model():
    """Helper to create a simple NPVModel for plotting tests."""
    co2_rv  = RandomVariable("co2_price",    "triangular", low=20,   mode=80,   high=200)
    elec_rv = RandomVariable("electricity",  "normal",     mean=0.12, std=0.03)
    mat_rv  = RandomVariable("raw_material", "lognormal",  mean=4.5,  std=0.3)
    model   = NPVModel(horizon_years=10, discount_rate=0.05)
    model.add_cost(CostComponent("CO2 cost",     quantity=5.0,  unit_price=co2_rv))
    model.add_cost(CostComponent("Electricity",  quantity=800,  unit_price=elec_rv))
    model.add_cost(CostComponent("Raw material", quantity=1.1,  unit_price=mat_rv))
    return model


# ─── plot_output_distribution ─────────────────────────────────────────────────

class TestPlotOutputDistribution:

    def setup_method(self):
        self.result = make_result()

    def teardown_method(self):
        plt.close("all")  # clean up figures after each test

    def test_returns_figure(self):
        fig = plotting.plot_output_distribution(self.result)
        assert isinstance(fig, plt.Figure)

    def test_correct_title(self):
        fig = plotting.plot_output_distribution(self.result, title="Test Title")
        assert fig.axes[0].get_title() == "Test Title"

    def test_correct_xlabel(self):
        fig = plotting.plot_output_distribution(self.result, xlabel="NPV (€)")
        assert fig.axes[0].get_xlabel() == "NPV (€)"

    def test_percentile_lines_shown(self):
        """When show_percentiles=True, there should be 3 vertical lines (p5, p95, mean)."""
        fig = plotting.plot_output_distribution(self.result, show_percentiles=True)
        vlines = [l for l in fig.axes[0].get_lines()]
        assert len(vlines) >= 1  # at least the mean line

    def test_no_percentile_lines(self):
        """When show_percentiles=False, no vertical lines should be drawn."""
        fig = plotting.plot_output_distribution(self.result, show_percentiles=False)
        lines = fig.axes[0].get_lines()
        assert len(lines) == 0

    def test_has_legend(self):
        fig = plotting.plot_output_distribution(self.result, show_percentiles=True)
        assert fig.axes[0].get_legend() is not None


# ─── plot_convergence ─────────────────────────────────────────────────────────

class TestPlotConvergence:

    def setup_method(self):
        model  = lambda s: s["co2_price"] * 5.0
        rv     = RandomVariable("co2_price", "triangular", low=20, mode=80, high=200)
        sim    = MonteCarloSimulator(model=model, n_samples=500, seed=0)
        sim.add_input(rv)
        self.conv = sim.run_convergence_study(
            sample_sizes=[10, 50, 100, 500],
            metric="mean",
            n_repetitions=3
        )

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plotting.plot_convergence(self.conv)
        assert isinstance(fig, plt.Figure)

    def test_xaxis_is_log_scaled(self):
        fig = plotting.plot_convergence(self.conv)
        assert fig.axes[0].get_xscale() == "log"

    def test_correct_xlabel(self):
        fig = plotting.plot_convergence(self.conv)
        assert "samples" in fig.axes[0].get_xlabel().lower()

    def test_has_recommended_n_line(self):
        """Should have at least one vertical line for recommended N."""
        fig = plotting.plot_convergence(self.conv)
        vlines = [l for l in fig.axes[0].get_lines() if len(l.get_xdata()) == 2]
        assert len(vlines) >= 1

    def test_has_legend(self):
        fig = plotting.plot_convergence(self.conv)
        assert fig.axes[0].get_legend() is not None


# ─── plot_tornado ─────────────────────────────────────────────────────────────

class TestPlotTornado:

    def setup_method(self):
        self.result = make_result()

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plotting.plot_tornado(self.result)
        assert isinstance(fig, plt.Figure)

    def test_correct_title(self):
        fig = plotting.plot_tornado(self.result, title="Sensitivity")
        assert fig.axes[0].get_title() == "Sensitivity"

    def test_number_of_bars_matches_inputs(self):
        """Number of bars should equal number of input variables."""
        fig = plotting.plot_tornado(self.result)
        bars = fig.axes[0].patches
        assert len(bars) == len(self.result.inputs)

    def test_correct_xlabel(self):
        fig = plotting.plot_tornado(self.result)
        assert "correlation" in fig.axes[0].get_xlabel().lower()


# ─── plot_input_vs_output ─────────────────────────────────────────────────────

class TestPlotInputVsOutput:

    def setup_method(self):
        self.result = make_result()

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plotting.plot_input_vs_output(self.result, "co2_price")
        assert isinstance(fig, plt.Figure)

    def test_correct_xlabel(self):
        fig = plotting.plot_input_vs_output(self.result, "electricity")
        assert fig.axes[0].get_xlabel() == "electricity"

    def test_correct_title(self):
        fig = plotting.plot_input_vs_output(self.result, "co2_price")
        assert "co2_price" in fig.axes[0].get_title()

    def test_invalid_input_name(self):
        with pytest.raises(ValueError, match="not found"):
            plotting.plot_input_vs_output(self.result, "nonexistent")

    def test_number_of_points(self):
        """Scatter plot should have exactly n_samples points."""
        fig = plotting.plot_input_vs_output(self.result, "co2_price")
        scatter = fig.axes[0].collections[0]
        assert len(scatter.get_offsets()) == 1000


# ─── plot_simulation_input ────────────────────────────────────────────────────

class TestPlotSimulationInput:

    def setup_method(self):
        self.result = make_result()

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plotting.plot_simulation_input(self.result, "co2_price")
        assert isinstance(fig, plt.Figure)

    def test_correct_title(self):
        fig = plotting.plot_simulation_input(self.result, "co2_price")
        assert "co2_price" in fig.axes[0].get_title()

    def test_correct_xlabel(self):
        fig = plotting.plot_simulation_input(self.result, "electricity")
        assert fig.axes[0].get_xlabel() == "electricity"

    def test_has_kde_line(self):
        """Should have at least one line for the KDE."""
        fig = plotting.plot_simulation_input(self.result, "co2_price")
        assert len(fig.axes[0].get_lines()) >= 1

    def test_invalid_input_name(self):
        with pytest.raises(ValueError, match="not found in simulation inputs"):
            plotting.plot_simulation_input(self.result, "nonexistent")

    def test_has_legend(self):
        fig = plotting.plot_simulation_input(self.result, "co2_price")
        assert fig.axes[0].get_legend() is not None
        
# ─── plot_cost_breakdown ──────────────────────────────────────────────────────

class TestPlotCostBreakdown:

    def setup_method(self):
        self.result = make_result()
        self.model  = make_model()

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plotting.plot_cost_breakdown(self.result, self.model)
        assert isinstance(fig, plt.Figure)

    def test_correct_title(self):
        fig = plotting.plot_cost_breakdown(self.result, self.model, title="Costs")
        assert fig.axes[0].get_title() == "Costs"

    def test_number_of_bars_matches_components(self):
        """Number of bars should equal number of cost components."""
        fig = plotting.plot_cost_breakdown(self.result, self.model)
        bars = fig.axes[0].patches
        assert len(bars) == len(self.model._costs)

    def test_percentages_sum_to_100(self):
        contributions = {}
        for component in self.model._costs:
            total = 0.0
            for t in range(self.model.horizon_years):
                mean_samples = {
                    name: float(np.mean(arr))
                    for name, arr in self.result.inputs.items()
                }
                total += component.annual_cost(t, mean_samples) / (1 + self.model.discount_rate) ** t
            contributions[component.name] = total
        total_cost = sum(contributions.values())
        percentages = sum((v / total_cost) * 100 for v in contributions.values())
        assert abs(percentages - 100.0) < 0.01